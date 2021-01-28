
#pragma once

#include "ethash_cuda_miner_kernel.h"

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#include <functional>

#define MAX_STREAMS 4
#define CU_TARGET_BATCH_TIME 0.9F  // seconds

namespace dev
{
namespace eth
{
class CUDAMiner : public Miner
{
public:
    CUDAMiner(unsigned _index, DeviceDescriptor& _device);
    ~CUDAMiner() override;

    static int getNumDevices();
    static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);

protected:
    bool miner_init_device() override;
    bool miner_init_epoch() override;
    void miner_kick() override;
    void miner_set_header(const h256& header) override;
    void miner_set_target(uint64_t target) override;
    void miner_clear_counts(uint32_t streamIdx) override;
    void miner_reset_device() override;
    void miner_search(uint32_t streamIdx, uint64_t start_nonce) override;
    void miner_sync(uint32_t streamIdx, Search_results& search_buf) override;
    void miner_get_block_sizes(Block_sizes& blks) override;

private:
    ::Search_results* m_search_buf[MAX_STREAMS];
    cudaStream_t m_streams[MAX_STREAMS];

    uint64_t m_allocated_memory_dag = 0; // dag_size is a uint64_t in EpochContext struct
    size_t m_allocated_memory_light_cache = 0;
};


}  // namespace eth
}  // namespace dev
