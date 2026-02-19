#pragma once
#include <cuda_runtime.h>
#include <vector_functions.h>
#include "corekit/utils/assert.hpp"
#include "corekit/types.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>

using uint  = unsigned int;
using uchar = unsigned char;

namespace corekit {
    namespace cuda {

        using namespace corekit::types;
        using namespace corekit::utils;

        inline void check_cuda(cudaError_t err = cudaGetLastError(), const Status&   message  = "<NO DESCRIPTION>",
                       const Location& location = Location::current()) {
            corecheck(err == cudaSuccess, message, location);
        }

        inline bool is_device_pointer(const void* ptr) {
            cudaPointerAttributes attributes;

            const cudaError_t result =
                cudaPointerGetAttributes(&attributes, ptr);

            if (result != cudaSuccess) {
                return false;
            }

            return attributes.type == cudaMemoryTypeDevice;
        }

        template <typename T>
        bool is_device_pointer(const T* ptr) {
            return is_device_pointer(static_cast<const void*>(ptr));
        }

        inline void log_stage(cudaStream_t                 stream,
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

    }  // namespace cuda

}  // namespace corekit
