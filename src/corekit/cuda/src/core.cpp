#include "corekit/cuda/core.hpp"

namespace corekit {
    namespace cuda {

        Logger Tracing::logger("Cuda");
        std::atomic<uint64_t> Tracing::total_cuda_mem{0};

        void Tracing::log_stage(cudaStream_t                 stream,
                    const char*                  label,
                    const std::function<void()>& work) {
            cudaEvent_t start = nullptr;
            cudaEvent_t stop  = nullptr;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, stream);
            work();
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "[timer] " << label << ": " << ms << " ms"
                      << std::endl;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        uint64_t Tracing::request_bytes(uint64_t bytes) {
            const uint64_t after = total_cuda_mem.fetch_add(
                                       bytes,
                                       std::memory_order_relaxed) +
                                   bytes;
            logger.debug() << " Requested " << bytes
                           << " bytes on device (total: " << after
                           << " bytes)";
            return after;
        }

        uint64_t Tracing::release_bytes(uint64_t bytes) {
            const uint64_t after = total_cuda_mem.fetch_sub(
                                       bytes,
                                       std::memory_order_relaxed) -
                                   bytes;
            logger.debug() << " Released " << bytes
                           << " bytes on device (total: " << after
                           << " bytes)";
            return after;
        }

        uint64_t Tracing::total_bytes() {
            return total_cuda_mem.load(std::memory_order_relaxed);
        }



        void check_cuda(const cudaError_t& err, const Location& location) {
            corecheck(err == cudaSuccess, cudaGetErrorString(err), location);
        }

        bool is_device_pointer(const void* ptr) {
            cudaPointerAttributes attributes;

            const cudaError_t result =
                cudaPointerGetAttributes(&attributes, ptr);

            if (result != cudaSuccess) {
                return false;
            }

            return attributes.type == cudaMemoryTypeDevice;
        }

        

        uint64_t total_cuda_bytes() {
            return Tracing::total_bytes();
        }

    }  // namespace cuda
}  // namespace corekit
