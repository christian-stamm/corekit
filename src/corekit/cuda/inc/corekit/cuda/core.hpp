#pragma once
#include <cuda_runtime.h>
#include <vector_functions.h>

#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "corekit/types.hpp"
#include "corekit/utils/assert.hpp"

using uint  = unsigned int;
using uchar = unsigned char;

namespace corekit {
    namespace cuda {

        using namespace corekit::types;
        using namespace corekit::utils;

        inline void check_cuda(cudaError_t     err      = cudaGetLastError(),
                               const Status&   message  = "<NO DESCRIPTION>",
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

        template <typename T>
        struct NvMem {
            using Ptr = std::shared_ptr<NvMem>;

            NvMem(uint64_t elems = 0) : elems(elems) {
                T* rawptr = nullptr;

                if (0 < elems) {
                    check_cuda(cudaMalloc(&rawptr, elems * sizeof(T)),
                               "Failed to allocate memory in NvMem");

                    std::cout << "Created Memory of size " << get_bytes()
                              << " bytes" << std::endl;

                    buffer.reset(rawptr, [elems = this->elems](T* p) {
                        if (p) {
                            std::cout << "Destroyed Memory of size "
                                      << (elems * sizeof(T)) << " bytes"
                                      << std::endl;

                            check_cuda(cudaFree(p),
                                       "Failed to free memory in NvMem");
                        }
                    });
                }
            }

            NvMem(const NvMem& other) = default;
            NvMem(NvMem&& other)      = default;

            NvMem& operator=(const NvMem& other) = default;
            NvMem& operator=(NvMem&& other)      = default;

            inline uint get_elems() const {
                return elems;
            }

            inline uint64_t get_bytes() const {
                return elems * sizeof(T);
            }

            inline T* ptr() const {
                return buffer.get();
            }

           private:
            uint64_t           elems;
            std::shared_ptr<T> buffer;
        };

    }  // namespace cuda

}  // namespace corekit
