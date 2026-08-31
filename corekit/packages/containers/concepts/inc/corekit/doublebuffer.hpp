#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "corekit/memory.hpp"
#include "corekit/result.hpp"

namespace corekit {

    class DoubleBuffer : private Memory<uint8_t> {
        public:

            using Ptr = std::shared_ptr<DoubleBuffer>;

            explicit DoubleBuffer(size_t size, bool aligned = true)
                : Memory<uint8_t>(2 * size, aligned)
                , half_size_(size) { }

            [[nodiscard]]
            size_t half_size() const noexcept {
                return half_size_;
            }

            [[nodiscard]]
            size_t total_size() const noexcept {
                return 2 * half_size_;
            }

            [[nodiscard]]
            uint8_t* dma_data() noexcept {
                return Memory<uint8_t>::data();
            }

            [[nodiscard]]
            const uint8_t* dma_data() const noexcept {
                return Memory<uint8_t>::data();
            }

            [[nodiscard]]
            std::span<uint8_t> half(size_t index) noexcept {
                return {dma_data() + ((index & 1u) * half_size_), half_size_};
            }

            [[nodiscard]]
            std::span<const uint8_t> half(size_t index) const noexcept {
                return {dma_data() + ((index & 1u) * half_size_), half_size_};
            }

            // Compatibility helpers for non-DMA users. The DMA path should use
            // half(index) directly so software state cannot drift from hardware.
            [[nodiscard]]
            std::span<const uint8_t> read() const noexcept {
                return half(foreground_index_.load(std::memory_order_acquire));
            }

            [[nodiscard]]
            std::span<uint8_t> write() noexcept {
                return half(foreground_index_.load(std::memory_order_acquire) ^ 1u);
            }

            VoidResult flip() noexcept {
                foreground_index_.fetch_xor(1u, std::memory_order_acq_rel);
                return VoidResult();
            }

        private:

            size_t               half_size_ = 0;
            std::atomic<uint8_t> foreground_index_{0};
    };

} // namespace corekit
