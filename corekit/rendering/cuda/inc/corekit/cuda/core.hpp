#pragma once
#include <cuda_runtime.h>
#include <vector_functions.h>

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "corekit/types.hpp"
#include "corekit/utils/assert.hpp"
#include "corekit/utils/logger.hpp"

using uint  = unsigned int;
using uchar = unsigned char;

namespace corekit {
    namespace cuda {

        using namespace corekit::types;
        using namespace corekit::utils;

        void check_cuda(const cudaError_t& err      = cudaGetLastError(),
                        const Location&    location = Location::current());

        bool is_device_pointer(const void* ptr);

        template <typename T>
        bool is_device_pointer(const T* ptr) {
            return is_device_pointer(static_cast<const void*>(ptr));
        }

        struct Tracing {
           public:
            Tracing() = delete;

            static void log_stage(cudaStream_t                 stream,
                                  const char*                  label,
                                  const std::function<void()>& work);

            static uint64_t request_bytes(uint64_t bytes);
            static uint64_t release_bytes(uint64_t bytes);
            static uint64_t total_bytes();

           private:
            static Logger                logger;
            static std::atomic<uint64_t> total_cuda_mem;
        };

        template <typename T>
        struct NvMem {
            using Ptr = std::shared_ptr<NvMem>;

            NvMem(uint64_t elems = 0) : elems(elems) {
                T* rawptr = nullptr;

                if (0 < elems) {
                    const uint64_t bytes = elems * sizeof(T);

                    check_cuda(cudaMalloc(&rawptr, bytes));
                    const uint64_t after_alloc = Tracing::request_bytes(bytes);

                    buffer.reset(rawptr, [this, bytes = bytes](T* p) {
                        if (p) {
                            check_cuda(cudaFree(p));
                            const uint64_t after_free =
                                Tracing::release_bytes(bytes);
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
