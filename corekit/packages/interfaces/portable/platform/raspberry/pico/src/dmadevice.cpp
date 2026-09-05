#include "corekit/dmadevice.hpp"

#include <hardware/dma.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>

#include "corekit/math.hpp"
#include "corekit/mutex.hpp"

constexpr uint IRQ_INDEX = 0;
constexpr uint IRQ_NUM   = DMA_IRQ_NUM(IRQ_INDEX);

constexpr uint MIN_RING_BITS = 1;
constexpr uint MAX_RING_BITS = 15;

__isr void shared_irq_callback();

namespace corekit::Dma {

    std::array<Handle, NUM_DMA_CHANNELS> handles{};
    Mutex                                claim_mutex;

    // -----------------------------------------------------------------
    // Transfer
    // -----------------------------------------------------------------

    Transfer::Transfer(uint                 channel,
                       const volatile void* originAddr,
                       AddrUpdt             originUpdate,
                       const volatile void* targetAddr,
                       AddrUpdt             targetUpdate,
                       uint32_t             burst_length,
                       int32_t              wrap_length,
                       XferSize             blockSize,
                       uint                 dreq,
                       bool                 byteswap,
                       bool                 sniff,
                       int                  chain,
                       Handle&&             handle)
        : originAddr(const_cast<volatile void*>(originAddr))
        , targetAddr(const_cast<volatile void*>(targetAddr))
        , encoding(dma_encode_transfer_count(burst_length))
        , channel(channel)
        , config(dma_channel_get_default_config(channel))  //
    {
        if (wrap_length != 0) {
            size_t wrapBytes = std::abs(wrap_length) * (1 << blockSize);
            size_t ringSize  = std::log2(wrapBytes);

            const bool overflow  = MAX_RING_BITS < ringSize;
            const bool underflow = MIN_RING_BITS > ringSize;
            const bool isPow2    = math::isPow2(wrapBytes);

            if (!overflow && !underflow && isPow2) {
                channel_config_set_ring(&config, 0 < wrap_length, ringSize);
            }
        }

        if (0 <= chain) {
            channel_config_set_chain_to(&config, chain);
        }

        handles[channel] = std::move(handle);

        channel_config_set_dreq(&config, dreq);
        channel_config_set_transfer_data_size(&config, blockSize);
        channel_config_set_read_increment(&config, originUpdate);
        channel_config_set_write_increment(&config, targetUpdate);
        channel_config_set_sniff_enable(&config, sniff);
        channel_config_set_irq_quiet(&config, false);
        channel_config_set_bswap(&config, byteswap);
        channel_config_set_enable(&config, true);
    }

    Transfer::~Transfer() {
        handles[channel] = nullptr;
    }

    // -----------------------------------------------------------------
    // Device
    // -----------------------------------------------------------------

    Device::Device(uint channel)
        : AsyncDevice<uint32_t>(
              std::format("DMA{}", channel),
              {&dma_channel_hw_addr(channel)->al2_write_addr_trig, DREQ_FORCE},
              {&dma_channel_hw_addr(channel)->al3_read_addr_trig, DREQ_FORCE})
        , channel(channel) {
        dma_channel_claim(channel);
        dma_irqn_set_channel_enabled(IRQ_INDEX, channel, true);
    }

    Device::~Device() {
        dma_channel_cleanup(channel);
        dma_channel_unclaim(channel);
    }

    VoidResult Device::on_load() {
        return VoidResult();
    }

    VoidResult Device::on_unload() {
        kill();
        return VoidResult();
    }

    Device::Ptr Device::request_unused() {
        std::lock_guard lock(claim_mutex);

        for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
            if (!dma_channel_is_claimed(channel)) {
                return std::make_shared<Device>(channel);
            }
        }

        return nullptr;
    }

    bool Device::busy() const {
        return dma_channel_is_busy(channel);
    }

    void Device::kill() const {
        if (busy()) {
            dma_channel_abort(channel);
        }
    }

    bool Device::process(Transfer::Ptr task) {
        if (busy() || !task) {
            return false;
        }

        if (task->channel != channel) {
            return false;
        }

        dma_channel_set_transfer_count(channel, task->encoding, false);
        dma_channel_set_read_addr(channel, task->originAddr, false);
        dma_channel_set_write_addr(channel, task->targetAddr, false);
        dma_channel_set_config(channel, &task->config, true);

        return true;
    }

    void Device::enableIRQ() {
        irq_set_exclusive_handler(IRQ_NUM, shared_irq_callback);
        irq_set_enabled(IRQ_NUM, true);
    }

    void Device::disableIRQ() {
        irq_set_enabled(IRQ_NUM, false);
        irq_remove_handler(IRQ_NUM, shared_irq_callback);
    }
};  // namespace corekit::Dma

void shared_irq_callback() {
    using namespace corekit::Dma;

    for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
        if (dma_irqn_get_channel_status(IRQ_INDEX, channel)) {
            const Handle& handle = handles[channel];

            if (handle) {
                handle(channel);
            }

            dma_irqn_acknowledge_channel(IRQ_INDEX, channel);
        }
    }
}
