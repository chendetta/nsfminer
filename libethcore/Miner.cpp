
#include "Miner.h"
#include "Farm.h"

namespace dev
{
namespace eth
{

FarmFace* FarmFace::m_this = nullptr;

DeviceDescriptor Miner::getDescriptor()
{
    return m_deviceDescriptor;
}

void Miner::setWork(WorkPackage const& _work)
{
    {
        lock_guard<mutex> l(miner_work_mutex);

        // Void work if this miner is paused
        if (paused())
            m_work.header = h256();
        else
            m_work = _work;
#ifdef DEV_BUILD
        m_workSwitchStart = chrono::steady_clock::now();
#endif
    }
    miner_kick();
}

void Miner::ReportSolution(const h256& header, uint64_t nonce)
{
    cnote << EthWhite << "Job: " << header.abridged()
          << " Solution: " << toHex(nonce, HexPrefix::Add);
}

void Miner::ReportDAGDone(uint64_t dagSize, uint32_t dagTime)
{
    cnote << dev::getFormattedMemory(float(dagSize)) << " of DAG data generated in " << fixed
          << setprecision(1) << dagTime / 1000.0f << " seconds";
}

void Miner::ReportGPUMemoryUsage(uint64_t requiredTotalMemory, uint64_t totalMemory)
{
    cnote << "Using " << dev::getFormattedMemory(float(requiredTotalMemory)) << " out of "
          << dev::getFormattedMemory(float(totalMemory)) << " GPU memory";
}

void Miner::ReportGPUNoMemoryAndPause(uint64_t requiredMemory, uint64_t totalMemory)
{
    cwarn << "Epoch " << m_epochContext.epochNumber << " requires "
          << dev::getFormattedMemory((double)requiredMemory) << " memory. Only "
          << dev::getFormattedMemory((double)totalMemory) << " available on device.";
    pause(MinerPauseEnum::PauseDueToInsufficientMemory);
}

void Miner::pause(MinerPauseEnum what) 
{
    lock_guard<mutex> l(x_pause);
    m_pauseFlags.set(what);
    m_work.header = h256();
    miner_kick();
}

bool Miner::paused()
{
    lock_guard<mutex> l(x_pause);
    return m_pauseFlags.any();
}

bool Miner::pauseTest(MinerPauseEnum what)
{
    lock_guard<mutex> l(x_pause);
    return m_pauseFlags.test(what);
}

string Miner::pausedString()
{
    lock_guard<mutex> l(x_pause);
    string retVar;
    if (m_pauseFlags.any())
    {
        for (int i = 0; i < MinerPauseEnum::Pause_MAX; i++)
        {
            if (m_pauseFlags[(MinerPauseEnum)i])
            {
                if (!retVar.empty())
                    retVar.append("; ");

                if (i == MinerPauseEnum::PauseDueToOverHeating)
                    retVar.append("Overheating");
                else if (i == MinerPauseEnum::PauseDueToAPIRequest)
                    retVar.append("Api request");
                else if (i == MinerPauseEnum::PauseDueToFarmPaused)
                    retVar.append("Farm suspended");
                else if (i == MinerPauseEnum::PauseDueToInsufficientMemory)
                    retVar.append("Insufficient GPU memory");
                else if (i == MinerPauseEnum::PauseDueToInitEpochError)
                    retVar.append("Epoch initialization error");

            }
        }
    }
    return retVar;
}

void Miner::resume(MinerPauseEnum fromwhat) 
{
    lock_guard<mutex> l(x_pause);
    m_pauseFlags.reset(fromwhat);
    //if (!m_pauseFlags.any())
    //{
    //    // TODO Push most recent job from farm ?
    //    // If we do not push a new job the miner will stay idle
    //    // till a new job arrives
    //}
}

float Miner::RetrieveHashRate() noexcept
{
    return m_hashRate.load(memory_order_relaxed);
}

void Miner::TriggerHashRateUpdate() noexcept
{
    bool b = false;
    if (m_hashRateUpdate.compare_exchange_weak(b, true))
        return;
    // GPU didn't respond to last trigger, assume it's dead.
    // This can happen on CUDA if:
    //   runtime of --cuda-grid-size * --cuda-streams exceeds time of m_collectInterval
    m_hashRate = 0.0;
}

WorkPackage Miner::work() const
{
    unique_lock<mutex> l(miner_work_mutex);
    return m_work;
}

void Miner::updateHashRate(uint32_t _groupSize, uint32_t _increment) noexcept
{
    m_groupCount += _increment * _groupSize;
    bool b = true;
    if (!m_hashRateUpdate.compare_exchange_weak(b, false))
        return;
    using namespace chrono;
    auto t = steady_clock::now();
    auto us = duration_cast<microseconds>(t - m_hashTime).count();
    m_hashTime = t;

    m_hashRate.store(us ? (float(m_groupCount) * 1.0e6f) / us : 0.0f, memory_order_relaxed);
    m_groupCount = 0;
}

void Miner::workLoop()
{
    if (!miner_init_device())
        return;

    m_gpuInitialized.store(true);

    try
    {
        while (!shouldStop())
        {
            const WorkPackage current(work());
            if (!current)
            {
                unique_lock<mutex> l(miner_work_mutex);
                // must be less than telemetry interval
                m_new_work_signal.wait_for(l, chrono::seconds(3));
                continue;
            }

            // Epoch change ?
            if (current.epoch != m_lastEpoch)
            {
                m_resourceInitialized.store(false);
                if (!miner_init_epoch())
                    break;
                m_resourceInitialized.store(true);
                m_lastEpoch = current.epoch;

                // As DAG generation takes a while we need to
                // ensure we're on latest job, not on the one
                // which triggered the epoch change
                continue;
            }

            // Eventually start searching
            search(current.header, uint64_t((u64)((u256)current.boundary >> 192)),
                current.startNonce, current);
        }

    }
    catch (const runtime_error& e)
    {
        throw runtime_error(string("GPU error: ") + e.what());
    }
    // Reset miner and stop working
    m_gpuInitialized = false;
    miner_reset_device();
}

void Miner::search(
    const h256& header, uint64_t target, uint64_t start_nonce, const dev::eth::WorkPackage& w)
{
    miner_set_header(header);
    if (m_current_target != target)
    {
        miner_set_target(target);
        m_current_target = target;
    }
    Block_sizes bs;
    miner_get_block_sizes(bs);
    const uint32_t stream_blocks(bs.multiplier * bs.block_size);
    const uint32_t batch_blocks(bs.streams * stream_blocks);

    // NOTE: The following struct must match the one defined in
    // ethash.cl
    struct Search_results search_buf[MAX_STREAMS];

    // prime each stream, clear search result buffers and start the search
    for (uint32_t streamIdx = 0; streamIdx < bs.streams; streamIdx++, start_nonce += stream_blocks)
    {
        miner_clear_counts(streamIdx);
        m_hung_miner.store(false);
        miner_search(streamIdx, start_nonce);
    }

    uint32_t streams_bsy((1 << bs.streams) - 1);

    // process stream batches until we get new work.

    while (streams_bsy)
    {
        uint32_t batchCount(0);

        // This inner loop will process each cuda stream individually
        for (uint32_t streamIdx = 0; streamIdx < bs.streams;
             streamIdx++, start_nonce += stream_blocks)
        {
            uint32_t stream_mask(1 << streamIdx);
            if (!(streams_bsy & stream_mask))
                continue;

            // Wait for this stream complete
            miner_sync(streamIdx, search_buf[streamIdx]);

            Search_results& r(search_buf[streamIdx]);

            if (!r.done)
                r.done = paused() || shouldStop();

            // clear solution count, hash count and done
            miner_clear_counts(streamIdx);

            batchCount += r.hashCount;

            if (r.done)
                streams_bsy &= ~stream_mask;
            else
            {
                m_hung_miner.store(false);
                miner_search(streamIdx, start_nonce);
            }

            if (r.solCount)
                for (uint32_t i = 0; i < r.solCount; i++)
                {
                    uint64_t nonce(start_nonce - batch_blocks + r.results[i].gid);
                    h256 mix((::byte*)&r.results[i].mix, h256::ConstructFromPointer);
                    Farm::f().submitProof(
                        Solution{nonce, mix, w, chrono::steady_clock::now(), m_index});
                    ReportSolution(w.header, nonce);
                }
        }
        updateHashRate(bs.block_size, batchCount);
    }

#ifdef DEV_BUILD
    // Optionally log job switch time
    if (!shouldStop() && (g_logOptions & LOG_SWITCH))
        cnote << "Switch time: "
              << chrono::duration_cast<chrono::microseconds>(
                     chrono::steady_clock::now() - m_workSwitchStart)
                     .count()
              << " us.";
#endif
}

}  // namespace eth
}  // namespace dev
