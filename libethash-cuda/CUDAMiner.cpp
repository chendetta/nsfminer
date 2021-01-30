
#include <libethcore/Farm.h>

#include "CUDAMiner.h"

using namespace std;
using namespace dev;
using namespace eth;

CUDAMiner::CUDAMiner(unsigned _index, DeviceDescriptor& _device) : Miner("cu-", _index)
{
    m_deviceDescriptor = _device;
    m_block_multiple = 1000;
}

CUDAMiner::~CUDAMiner()
{
    stopWorking();
    miner_kick();
}

#define HostToDevice(dst, src, siz) CUDA_CALL(cudaMemcpy(dst, src, siz, cudaMemcpyHostToDevice))

#define DeviceToHost(dst, src, siz) CUDA_CALL(cudaMemcpy(dst, src, siz, cudaMemcpyDeviceToHost))

bool CUDAMiner::miner_init_device()
{
    cnote << "Using Pci " << m_deviceDescriptor.uniqueId << ": " << m_deviceDescriptor.cuName
          << " (Compute " + m_deviceDescriptor.cuCompute + ") Memory : "
          << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory);

    // Set Hardware Monitor Info
    m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
    m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
    m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)

    try
    {
        CUDA_CALL(cudaSetDevice(m_deviceDescriptor.cuDeviceIndex));
        CUDA_CALL(cudaDeviceReset());
    }
    catch (const runtime_error& ec)
    {
        cnote << "Could not set CUDA device on Pci Id " << m_deviceDescriptor.uniqueId
              << " Error : " << ec.what();
        cnote << "Mining aborted on this device.";
        return false;
    }
    return true;
}

bool CUDAMiner::miner_init_epoch()
{
    // If we get here it means epoch has changed so it's not necessary
    // to check again dag sizes. They're changed for sure
    m_current_target = 0;
    auto startInit = chrono::steady_clock::now();
    size_t RequiredTotalMemory = (m_epochContext.dagSize + m_epochContext.lightSize);

    try
    {
        hash128_t* dag;
        hash64_t* light;

        // If we have already enough memory allocated, we just have to
        // copy light_cache and regenerate the DAG
        if (m_allocated_memory_dag < m_epochContext.dagSize ||
            m_allocated_memory_light_cache < m_epochContext.lightSize)
        {
            // We need to reset the device and (re)create the dag
            // cudaDeviceReset() frees all previous allocated memory
            CUDA_CALL(cudaDeviceReset());
            CUDA_CALL(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
            CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

            // Check whether the current device has sufficient memory every time we recreate the dag
            if (m_deviceDescriptor.totalMemory < RequiredTotalMemory)
            {
                ReportGPUNoMemoryAndPause(RequiredTotalMemory, m_deviceDescriptor.totalMemory);
                return false;  // This will prevent to exit the thread and
                               // Eventually resume mining when changing coin or epoch (NiceHash)
            }
            // Release the pause flag if any
            resume(MinerPauseEnum::PauseDueToInsufficientMemory);
            resume(MinerPauseEnum::PauseDueToInitEpochError);

            // create buffer for cache
            CUDA_CALL(cudaMalloc((void**)&light, m_epochContext.lightSize));
            m_allocated_memory_light_cache = m_epochContext.lightSize;
            CUDA_CALL(cudaMalloc((void**)&dag, m_epochContext.dagSize));
            m_allocated_memory_dag = m_epochContext.dagSize;

            // create mining buffers
            for (unsigned i = 0; i < m_deviceDescriptor.cuStreamSize; ++i)
            {
                CUDA_CALL(cudaMalloc(&m_search_buf[i], sizeof(Search_results)));
                CUDA_CALL(cudaStreamCreateWithFlags(&m_streams[i], cudaStreamNonBlocking));
            }
        }
        else
            get_constants(&dag, NULL, &light, NULL);

        ReportGPUMemoryUsage(RequiredTotalMemory, m_deviceDescriptor.totalMemory);


        HostToDevice(light, m_epochContext.lightCache, m_epochContext.lightSize);

        set_constants(dag, m_epochContext.dagNumItems, light,
            m_epochContext.lightNumItems);  // in ethash_cuda_miner_kernel.cu

        ethash_generate_dag(
            m_epochContext.dagSize, m_block_multiple, m_deviceDescriptor.cuBlockSize, m_streams[0]);

        ReportDAGDone(m_allocated_memory_dag, uint32_t(chrono::duration_cast<chrono::milliseconds>(
                                                  chrono::steady_clock::now() - startInit)
                                                           .count()));
    }
    catch (const runtime_error& ec)
    {
        cnote << "Unexpected error " << ec.what() << " on CUDA device "
              << m_deviceDescriptor.uniqueId;
        cnote << "Mining suspended ...";
        pause(MinerPauseEnum::PauseDueToInitEpochError);
        return false;
    }

    return true;
}

static uint32_t one = 1;

void CUDAMiner::miner_kick()
{
    if (resourceInitialized() && gpuInitialized())
    {
        for (unsigned i = 0; i < m_deviceDescriptor.cuStreamSize; i++)
            CUDA_CALL(cudaMemcpyAsync((uint8_t*)m_search_buf[i] + offsetof(Search_results, done),
                &one, sizeof(one), cudaMemcpyHostToDevice));
    }
}

int CUDAMiner::getNumDevices()
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess)
        return deviceCount;

    if (err == cudaErrorInsufficientDriver)
    {
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        if (driverVersion == 0)
            cwarn << "No CUDA driver found";
        else
            cwarn << "Insufficient CUDA driver " << to_string(driverVersion);
    }
    else
        ccrit << "CUDA Error : " << cudaGetErrorString(err);

    return 0;
}

void CUDAMiner::enumDevices(map<string, DeviceDescriptor>& _DevicesCollection)
{
    int numDevices = getNumDevices();

    for (int i = 0; i < numDevices; i++)
    {
        string uniqueId;
        ostringstream s;
        DeviceDescriptor deviceDescriptor;
        cudaDeviceProp props;

        try
        {
            size_t freeMem, totalMem;
            CUDA_CALL(cudaGetDeviceProperties(&props, i));
            CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));
            s << "????:" << setfill('0') << hex << setw(2) << props.pciBusID << ':' << setw(2)
              << props.pciDeviceID << ".0";
            if (_DevicesCollection.find(s.str()) != _DevicesCollection.end())
                _DevicesCollection.erase(s.str());
            s.str("");
            s << setw(4) << setfill('0') << hex << props.pciDomainID << ':' << setw(2)
              << props.pciBusID << ':' << setw(2) << props.pciDeviceID << ".0";
            uniqueId = s.str();

            if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
                deviceDescriptor = _DevicesCollection[uniqueId];
            else
                deviceDescriptor = DeviceDescriptor();

            deviceDescriptor.name = string(props.name);
            deviceDescriptor.cuDetected = true;
            deviceDescriptor.uniqueId = uniqueId;
            deviceDescriptor.type = DeviceTypeEnum::Gpu;
            deviceDescriptor.cuDeviceIndex = i;
            deviceDescriptor.cuDeviceOrdinal = i;
            deviceDescriptor.cuName = string(props.name);
            deviceDescriptor.totalMemory = totalMem;
            deviceDescriptor.cuCompute =
                (to_string(props.major) + "." + to_string(props.minor));
            deviceDescriptor.cuComputeMajor = props.major;
            deviceDescriptor.cuComputeMinor = props.minor;
            deviceDescriptor.cuBlockSize = 128;
            deviceDescriptor.cuStreamSize = 2;


            _DevicesCollection[uniqueId] = deviceDescriptor;
        }
        catch (const runtime_error& _e)
        {
            ccrit << _e.what();
        }
    }
}

#if 0
static const uint32_t zero3[3] = {0, 0, 0};  // zero the result count

void CUDAMiner::search(
    uint8_t const* header, uint64_t target, uint64_t start_nonce, const dev::eth::WorkPackage& w)
{
    set_header(*((const hash32_t*)header));
    if (m_current_target != target)
    {
        set_target(target);
        m_current_target = target;
    }
    uint32_t batch_blocks(m_block_multiple * m_deviceDescriptor.cuBlockSize);
    uint32_t stream_blocks(batch_blocks * m_deviceDescriptor.cuStreamSize);

    // prime each stream, clear search result buffers and start the search
    for (uint32_t streamIdx = 0; streamIdx < m_deviceDescriptor.cuStreamSize;
         streamIdx++, start_nonce += batch_blocks)
    {
        HostToDevice(m_search_buf[streamIdx], zero3, sizeof(zero3));
        m_hung_miner.store(false);
        run_ethash_search(m_block_multiple, m_deviceDescriptor.cuBlockSize, m_streams[streamIdx],
            m_search_buf[streamIdx], start_nonce);
    }

    m_done = false;
    uint32_t streams_bsy((1 << m_deviceDescriptor.cuStreamSize) - 1);

    // process stream batches until we get new work.

    while (streams_bsy)
    {
        if (!m_done)
            m_done = paused();

        uint32_t batchCount(0);

        // This inner loop will process each cuda stream individually
        for (uint32_t streamIdx = 0; streamIdx < m_deviceDescriptor.cuStreamSize;
             streamIdx++, start_nonce += batch_blocks)
        {
            uint32_t stream_mask(1 << streamIdx);
            if (!(streams_bsy & stream_mask))
                continue;

            cudaStream_t stream(m_streams[streamIdx]);
            uint8_t* buffer((uint8_t*)m_search_buf[streamIdx]);

            // Wait for the stream complete
            CUDA_CALL(cudaStreamSynchronize(stream));

            Search_results r;

            DeviceToHost(&r.counts, buffer + offsetof(Search_results, counts), sizeof(r.counts));

            // clear solution count, hash count and done
            HostToDevice(buffer, zero3, sizeof(zero3));

            r.counts.solCount = min(r.counts.solCount, MAX_SEARCH_RESULTS);
            batchCount += r.counts.hashCount;

            if (r.counts.solCount)
                DeviceToHost(&r.results, buffer + offsetof(Search_results, results),
                    r.counts.solCount * sizeof(Search_Result));

            if (m_done)
                streams_bsy &= ~stream_mask;
            else
            {
                m_hung_miner.store(false);
                run_ethash_search(m_block_multiple, m_deviceDescriptor.cuBlockSize, stream,
                    (Search_results*)buffer, start_nonce);
            }

            if (r.counts.solCount)
                for (uint32_t i = 0; i < r.counts.solCount; i++)
                {
                    uint64_t nonce(start_nonce - stream_blocks + r.results[i].gid);
                    h256 mix((::byte*)&r.results[i].mix, h256::ConstructFromPointer);

                    Farm::f().submitProof(
                        Solution{nonce, mix, w, chrono::steady_clock::now(), m_index});
                    ReportSolution(w.header, nonce);
                }
            if (shouldStop())
                m_done = true;
        }
        updateHashRate(m_deviceDescriptor.cuBlockSize, batchCount);
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
#endif

static uint32_t zeros[3] = {0, 0, 0};

void CUDAMiner::miner_clear_counts(uint32_t streamIdx)
{
    // clear solution count, hash count and done
    HostToDevice(m_search_buf[streamIdx], zeros, sizeof(zeros));
}

void CUDAMiner::miner_reset_device()
{
    CUDA_CALL(cudaDeviceReset());
}

void CUDAMiner::miner_search(uint32_t streamIdx, uint64_t start_nonce)
{
    m_hung_miner.store(false);
    run_ethash_search(m_block_multiple, m_deviceDescriptor.cuBlockSize, m_streams[streamIdx],
        m_search_buf[streamIdx], start_nonce);
}

void CUDAMiner::miner_sync(uint32_t streamIdx, Search_results& results)
{
    // Wait for the stream complete
    CUDA_CALL(cudaStreamSynchronize(m_streams[streamIdx]));
    DeviceToHost(&results, m_search_buf[streamIdx], 3 * sizeof(uint32_t));

    if (results.solCount > MAX_SEARCH_RESULTS)
        results.solCount = MAX_SEARCH_RESULTS;

    if (results.solCount)
        DeviceToHost(results.results,
            (uint8_t*)m_search_buf[streamIdx] + offsetof(Search_results, results),
            results.solCount * sizeof(Search_result));
}

void CUDAMiner::miner_set_header(const h256& header)
{
    set_header(*((const hash32_t*)header.data()));
}

void CUDAMiner::miner_set_target(uint64_t target)
{
    set_target(target);
}

void CUDAMiner::miner_get_block_sizes(Block_sizes& blks)
{
    float hr = RetrieveHashRate();
    if (hr >= 1e7)
        m_block_multiple =
            uint32_t((hr * CU_TARGET_BATCH_TIME) /
                     (m_deviceDescriptor.cuStreamSize * m_deviceDescriptor.cuBlockSize));
    blks.streams = m_deviceDescriptor.cuStreamSize;
    blks.block_size = m_deviceDescriptor.cuBlockSize;
    blks.multiplier = m_block_multiple;
}
