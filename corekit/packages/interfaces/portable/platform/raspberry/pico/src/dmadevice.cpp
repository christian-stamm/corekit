#include "corekit/dmadevice.hpp"

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

namespace corekit::Dma {

    std::vector<Handle> handles(NUM_DMA_CHANNELS, nullptr);

    // -----------------------------------------------------------------
    // Transfer
    // -----------------------------------------------------------------

    Transfer::Transfer(uint                 channel,
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
                       Handle&&             handle)
        : originAddr(const_cast<volatile void*>(originAddr))
        , targetAddr(const_cast<volatile void*>(targetAddr))
        , channel(channel)  //
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

    Transfer::~Transfer() {
        handles[channel] = nullptr;
    }

    template <typename T>
    Transfer::Ptr Transfer::mem2dev(uint               chn,
                                    std::span<const T> src,
                                    const CtrlBlock&   dst,
                                    bool               repeat,
                                    bool               reverse,
                                    bool               byteswap,
                                    bool               sniff,
                                    int                chain,
                                    Handle&&           handle) {
        XferSize       blockSize = static_cast<XferSize>(sizeof(T) >> 1);
        volatile void* srcAddr   = reinterpret_cast<volatile void*>(
            const_cast<T*>(reverse ? &src.back() : &src.front()));

        return std::make_shared<Transfer>(
            chn,
            dst.dreq,
            srcAddr,
            dst.addr,
            reverse ? DMA_ADDRESS_UPDATE_DECREMENT
                    : DMA_ADDRESS_UPDATE_INCREMENT,
            DMA_ADDRESS_UPDATE_NONE,
            src.size(),
            blockSize,
            repeat ? Wrapping::Read : Wrapping::None,
            byteswap,
            sniff,
            chain,
            std::move(handle));
    }

    template <typename T>
    Transfer::Ptr Transfer::dev2mem(uint             chn,
                                    const CtrlBlock& src,
                                    std::span<T>     dst,
                                    bool             repeat,
                                    bool             reverse,
                                    bool             byteswap,
                                    bool             sniff,
                                    int              chain,
                                    Handle&&         handle) {
        XferSize       blockSize = static_cast<XferSize>(sizeof(T) >> 1);
        volatile void* dstAddr   = reinterpret_cast<volatile void*>(
            reverse ? &dst.back() : &dst.front());

        return std::make_shared<Transfer>(
            chn,
            src.dreq,
            src.addr,
            dstAddr,
            DMA_ADDRESS_UPDATE_NONE,
            reverse ? DMA_ADDRESS_UPDATE_DECREMENT
                    : DMA_ADDRESS_UPDATE_INCREMENT,
            dst.size(),
            blockSize,
            repeat ? Wrapping::Write : Wrapping::None,
            byteswap,
            sniff,
            chain,
            std::move(handle));
    }

    template <typename T>
    Transfer::Ptr Transfer::mem2mem(uint               chn,
                                    std::span<const T> src,
                                    std::span<T>       dst,
                                    bool               reverse,
                                    bool               byteswap,
                                    Handle&&           handle) {
        XferSize blockSize = static_cast<XferSize>(sizeof(T) >> 1);
        return nullptr;
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

    bool Device::on_load() {
        return true;
    }

    bool Device::on_unload() {
        kill();
        return true;
    }

    Device::Ptr Device::requestUnused() {
        for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
            if (!dma_channel_is_claimed(channel)) {
                return std::make_shared<Device>(channel);
            }
        }

        throw std::runtime_error("No unused Dma channels available.");
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

    template Transfer::Ptr Transfer::mem2dev<uint8_t>(  //
        uint,
        std::span<const uint8_t>,
        const CtrlBlock&,
        bool,
        bool,
        bool,
        bool,
        int,
        Handle&&  //
    );

    template Transfer::Ptr Transfer::mem2dev<uint16_t>(  //
        uint,
        std::span<const uint16_t>,
        const CtrlBlock&,
        bool,
        bool,
        bool,
        bool,
        int,
        Handle&&  //
    );

    template Transfer::Ptr Transfer::mem2dev<uint32_t>(  //
        uint,
        std::span<const uint32_t>,
        const CtrlBlock&,
        bool,
        bool,
        bool,
        bool,
        int,
        Handle&&  //
    );

    template Transfer::Ptr Transfer::dev2mem<uint8_t>(  //
        uint,
        const CtrlBlock&,
        std::span<uint8_t>,
        bool,
        bool,
        bool,
        bool,
        int,
        Handle&&  //
    );

    template Transfer::Ptr Transfer::dev2mem<uint16_t>(  //
        uint,
        const CtrlBlock&,
        std::span<uint16_t>,
        bool,
        bool,
        bool,
        bool,
        int,
        Handle&&  //
    );

    template Transfer::Ptr Transfer::dev2mem<uint32_t>(  //
        uint,
        const CtrlBlock&,
        std::span<uint32_t>,
        bool,
        bool,
        bool,
        bool,
        int,
        Handle&&  //
    );

    template Transfer::Ptr Transfer::mem2mem<uint8_t>(  //
        uint,
        std::span<const uint8_t>,
        std::span<uint8_t>,
        bool,
        bool,
        Handle&&  //
    );

    template Transfer::Ptr Transfer::mem2mem<uint16_t>(  //
        uint,
        std::span<const uint16_t>,
        std::span<uint16_t>,
        bool,
        bool,
        Handle&&  //
    );

    template Transfer::Ptr Transfer::mem2mem<uint32_t>(  //
        uint,
        std::span<const uint32_t>,
        std::span<uint32_t>,
        bool,
        bool,
        Handle&&  //
    );

};  // namespace corekit::Dma

void shared_irq_callback() {
    using namespace corekit::Dma;

    for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
        if (dma_irqn_get_channel_status(IRQ_INDEX, channel)) {
            const Handle handle = handles[channel];

            if (handle) {
                handle();
            }

            dma_irqn_acknowledge_channel(IRQ_INDEX, channel);
        }
    }
}
