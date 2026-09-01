#include "corekit/platform/dmadevice.hpp"

#include <hardware/dma.h>

#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <vector>

constexpr uint IRQ_INDEX = 0;
constexpr uint IRQ_NUM   = DMA_IRQ_NUM(IRQ_INDEX);

constexpr uint MIN_RING_BITS = 1;
constexpr uint MAX_RING_BITS = 15;

__isr void shared_irq_callback();

namespace corekit::platform {

    std::vector<Dma::Handle> handles(NUM_DMA_CHANNELS, nullptr);

    // -----------------------------------------------------------------
    // Transfer
    // -----------------------------------------------------------------

    Dma::Transfer::Transfer(uint                 channel,
                            uint                 dreq,
                            const volatile void* originAddr,
                            const volatile void* targetAddr,
                            AddrUpdt             originUpdate,
                            AddrUpdt             targetUpdate,
                            uint32_t             length,
                            XferSize             blockSize,
                            Wrapping             wrapping,
                            bool                 byteswap,
                            bool                 sniff,
                            int                  chain,
                            Dma::Handle&&        handle)
        : originAddr(const_cast<volatile void*>(originAddr))
        , targetAddr(const_cast<volatile void*>(targetAddr))  //
    {
        config   = dma_channel_get_default_config(channel);
        encoding = dma_encode_transfer_count(length);

        if (wrapping != Wrapping::None) {
            size_t wrapBytes = length * (1 << blockSize);
            size_t ringSize  = std::log2(wrapBytes);

            const bool overflow  = MAX_RING_BITS < ringSize;
            const bool underflow = MIN_RING_BITS > ringSize;
            const bool isPow2    = (wrapBytes & (wrapBytes - 1)) == 0;

            const bool           wrapWrite = wrapping == Wrapping::Write;
            const volatile void* ptr = wrapWrite ? targetAddr : originAddr;

            const bool isAligned =
                (reinterpret_cast<uintptr_t>(ptr) % wrapBytes) == 0;

            if (!overflow && !underflow && isPow2 && isAligned) {
                channel_config_set_ring(&config, wrapWrite, ringSize);
            }

            encoding = dma_encode_transfer_count_with_self_trigger(length);
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

    // -----------------------------------------------------------------
    // DmaDevice
    // -----------------------------------------------------------------

    DmaDevice::DmaDevice(uint channel)
        : AsyncDevice<uint32_t>(
              std::format("DMA{}", channel),
              {&dma_channel_hw_addr(channel)->al2_write_addr_trig, DREQ_FORCE},
              {&dma_channel_hw_addr(channel)->al3_read_addr_trig, DREQ_FORCE})
        , channel(channel) {
        dma_channel_claim(channel);
        dma_irqn_set_channel_enabled(IRQ_INDEX, channel, true);
    }

    DmaDevice::~DmaDevice() {
        dma_channel_cleanup(channel);
        dma_channel_unclaim(channel);
    }

    bool DmaDevice::on_load() {
        return true;
    }

    bool DmaDevice::on_unload() {
        kill();
        return true;
    }

    DmaDevice::Ptr DmaDevice::requestUnused() {
        for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
            if (!dma_channel_is_claimed(channel)) {
                return std::make_shared<DmaDevice>(channel);
            }
        }

        throw std::runtime_error("No unused Dma channels available.");
    }

    bool DmaDevice::busy() const {
        return dma_channel_is_busy(channel);
    }

    void DmaDevice::kill() const {
        if (busy()) {
            dma_channel_abort(channel);
        }
    }

    // void DMA::setMode(Mode mode, uint dreq, bool reversed, bool byteswap)
    // {
    //     AddrUpdt writeUpdate = DMA_ADDRESS_UPDATE_NONE;
    //     AddrUpdt readUpdate  = DMA_ADDRESS_UPDATE_NONE;

    //     if (mode == MEM2DEV || mode == MEM2MEM) {
    //         readUpdate = reversed ? DMA_ADDRESS_UPDATE_DECREMENT
    //                               : DMA_ADDRESS_UPDATE_INCREMENT;
    //     }

    //     if (mode == DEV2MEM || mode == MEM2MEM) {
    //         writeUpdate = reversed ? DMA_ADDRESS_UPDATE_DECREMENT
    //                                : DMA_ADDRESS_UPDATE_INCREMENT;
    //     }

    // }

    bool DmaDevice::process(Dma::Transfer::Ptr task) {
        if (busy() || !task) {
            return false;
        }

        dma_channel_set_transfer_count(channel, task->encoding, false);
        dma_channel_set_read_addr(channel, task->originAddr, false);
        dma_channel_set_write_addr(channel, task->targetAddr, false);
        dma_channel_set_config(channel, &task->config, true);

        return true;
    }

    void DmaDevice::enableIRQ() {
        irq_set_exclusive_handler(IRQ_NUM, shared_irq_callback);
        irq_set_enabled(IRQ_NUM, true);
    }

    void DmaDevice::disableIRQ() {
        irq_set_enabled(IRQ_NUM, false);
        irq_remove_handler(IRQ_NUM, shared_irq_callback);
    }

};  // namespace corekit::platform

void shared_irq_callback() {
    using namespace corekit::platform;

    for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
        if (dma_irqn_get_channel_status(IRQ_INDEX, channel)) {
            const Dma::Handle handle = handles[channel];

            if (handle) {
                handle();
            }

            dma_irqn_acknowledge_channel(IRQ_INDEX, channel);
        }
    }
}
