#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "corekit/check.hpp"
#include "corekit/heapbuffer.hpp"

namespace corekit {

    template <typename T>
    class DoubleBuffer {
            static constexpr long internal_instances = 3;

        public:

            using Ptr = std::shared_ptr<DoubleBuffer<T>>;

            explicit DoubleBuffer(std::size_t full_size)
                : full_(full_size)
                , ping_(full_.view(0, full_size / 2))
                , pong_(full_.view(full_size / 2, full_size / 2)) {
                core::check(full_size % 2 == 0, RuntimeError("double buffer size must be even"));
            }

            DoubleBuffer(const DoubleBuffer&) = delete;
            DoubleBuffer(DoubleBuffer&&)      = default;

            DoubleBuffer& operator=(const DoubleBuffer&) = delete;
            DoubleBuffer& operator=(DoubleBuffer&&)      = default;

            [[nodiscard]]
            std::size_t size() const noexcept {
                return ping_.size();
            }

            [[nodiscard]]
            HeapBuffer<const T> read() const noexcept {
                return ping_.as_const();
            }

            [[nodiscard]]
            HeapBuffer<T> write() noexcept {
                return pong_;
            }

            void flip() {
                const bool unused = full_.instances() == internal_instances;
                core::check(unused, RuntimeError("buffer is still in use"));
                if (core::ok()) { std::swap(ping_, pong_); }
            }

        private:

            HeapBuffer<T> full_;
            HeapBuffer<T> ping_;
            HeapBuffer<T> pong_;
    };

} // namespace corekit