#include "corekit/dmadevice.hpp"

#include <hardware/dma.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>

#include "corekit/atomic.hpp"
#include "corekit/error.hpp"
#include "corekit/math.hpp"
#include "corekit/result.hpp"
#include "corekit/semaphore.hpp"

namespace corekit::Dma {

    constexpr uint IRQ_INDEX     = 0;
    constexpr uint IRQ_NUM       = DMA_IRQ_NUM(IRQ_INDEX);

    constexpr uint MIN_RING_BITS = 1;
    constexpr uint MAX_RING_BITS = 15;

    std::array<Handle, NUM_DMA_CHANNELS>           irq_handles{};
    Semaphore                                      irq_trigger{0, 1};
    std::array<Atomic<uint32_t>, NUM_DMA_CHANNELS> irq_counts;
    Atomic<bool>                                   cb_modified{false};

    extern "C" {

    __isr void shared_dma_irq_callback() {
        const uint32_t chnmsk = dma_hw->ints0;
        dma_hw->ints0         = chnmsk; // Clear the interrupt flags

        uint32_t pending      = chnmsk;
        while (pending != 0) {
            const uint channel = static_cast<uint>(std::countr_zero(pending));
            irq_counts[channel].fetch_add(1);
            pending &= pending - 1;
        }

        irq_trigger.release();
    }
    }

    void enableIRQ() {
        irq_set_enabled(IRQ_NUM, false);

        for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
            irq_counts[channel].store(0);
            dma_irqn_acknowledge_channel(IRQ_INDEX, channel);
        }

        irq_set_exclusive_handler(IRQ_NUM, shared_dma_irq_callback);
        irq_set_enabled(IRQ_NUM, true);
    }

    void disableIRQ() {
        irq_set_enabled(IRQ_NUM, false);
        irq_remove_handler(IRQ_NUM, shared_dma_irq_callback);

        for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
            irq_counts[channel].store(0);
            dma_irqn_acknowledge_channel(IRQ_INDEX, channel);
            irq_handles[channel] = nullptr;
        }
    }

    // -----------------------------------------------------------------
    // IsrDaemon
    // -----------------------------------------------------------------

    IsrDaemon::IsrDaemon()
        : Task("DmaDaemon") { }

    void IsrDaemon::spin_once() const { irq_trigger.release(); }

    bool IsrDaemon::configure(uint channel, Handle handle) {
        corecheck(channel < NUM_DMA_CHANNELS, OutOfRangeError(std::format("Invalid DMA channel: {} (max={})", channel, NUM_DMA_CHANNELS - 1)));

        irq_handles[channel] = std::move(handle);
        cb_modified.store(true);
        return true;
    }

    void IsrDaemon::reconfigure() {
        if (!cb_modified.exchange(false)) { return; }

        for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
            const bool enabled = (irq_handles[channel] != nullptr);
            dma_irqn_set_channel_enabled(IRQ_INDEX, channel, false);
            dma_irqn_acknowledge_channel(IRQ_INDEX, channel);
            dma_irqn_set_channel_enabled(IRQ_INDEX, channel, enabled);
        }
    }

    VoidResult IsrDaemon::on_init(StopToken token) {
        enableIRQ();

        cb_modified.store(true);
        this->reconfigure();

        return VoidResult();
    }

    VoidResult IsrDaemon::on_exec(StopToken token) {
        while (!token.stop_requested()) {
            irq_trigger.acquire();
            this->reconfigure();

            for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
                const uint32_t count = irq_counts[channel].exchange(0);
                if (count == 0) { continue; }

                const Handle& handle = irq_handles[channel];
                if (!handle) { continue; }

                for (uint32_t n = 0; n < count; ++n) { handle(channel); }
            }
        }

        return VoidResult();
    }

    VoidResult IsrDaemon::on_exit(StopToken token) {
        disableIRQ();
        return VoidResult();
    }

    // -----------------------------------------------------------------
    // Transfer
    // -----------------------------------------------------------------

    Transfer::Transfer(
        uint                 channel,
        const volatile void* originAddr,
        AddrUpdt             originUpdate,
        const volatile void* targetAddr,
        AddrUpdt             targetUpdate,
        uint32_t             burst_length,
        int32_t              wrap_length,
        XferSize             blockSize,
        uint                 dreq,
        bool                 repeat,
        bool                 byteswap,
        bool                 sniff,
        int                  chain)
        : originAddr(const_cast<volatile void*>(originAddr))
        , targetAddr(const_cast<volatile void*>(targetAddr))
        , encoding(dma_encode_transfer_count(burst_length))
        , channel(channel)
        , config(dma_channel_get_default_config(channel))
    //
    {
        if (repeat) { encoding = dma_encode_transfer_count_with_self_trigger(burst_length); }

        if (wrap_length != 0) {
            const size_t transferBytes = size_t{1} << static_cast<uint>(blockSize);
            const size_t wrapBytes     = static_cast<size_t>(std::abs(wrap_length)) * transferBytes;

            corecheck(math::isPow2(wrapBytes), InvalidArgumentError("DMA ring size must be a power of two"));

            const uint ringBits = static_cast<uint>(std::countr_zero(wrapBytes));
            corecheck(ringBits >= MIN_RING_BITS && ringBits <= MAX_RING_BITS, OutOfRangeError("DMA ring size is outside the supported range"));

            const volatile void* ringBase = 0 < wrap_length ? targetAddr : originAddr;
            corecheck(ringBase != nullptr, InvalidArgumentError("DMA ring base address is null"));
            corecheck((reinterpret_cast<uintptr_t>(ringBase) % wrapBytes) == 0, InvalidArgumentError("DMA ring base address is not naturally aligned"));

            channel_config_set_ring(&config, 0 < wrap_length, ringBits);
        }

        if (0 <= chain) { channel_config_set_chain_to(&config, chain); }

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
    // Device
    // -----------------------------------------------------------------

    Device::Device()
        : Device::Device(dma_claim_unused_channel(true)){};

    Device::Device(uint channel)
        : AsyncDevice<uint32_t>(std::format("DMA{}", channel), {&dma_channel_hw_addr(channel)->al2_write_addr_trig, DREQ_FORCE}, {&dma_channel_hw_addr(channel)->al3_read_addr_trig, DREQ_FORCE})
        , channel(channel)
        , current_task(nullptr) { }

    Device::~Device() {
        dma_channel_cleanup(channel);
        dma_channel_unclaim(channel);
    }

    bool Device::on_load() { return true; }

    bool Device::on_unload() {
        kill();
        return true;
    }

    Device::Ptr Device::request_unused() { return std::make_shared<Device>(); }

    bool Device::busy() const { return dma_channel_is_busy(channel); }

    void Device::kill() const {
        if (busy()) { dma_channel_abort(channel); }
    }

    bool Device::configure(Transfer::Ptr task) {
        corecheck(task != nullptr, InvalidArgumentError("DMA Task is null"));
        corecheck(task->channel == channel, RuntimeError(std::format("DMA Task channel mismatch: task={} device={}", task->channel, channel)));

        this->kill();
        current_task = std::move(task);

        irq_counts[channel].store(0);
        dma_irqn_acknowledge_channel(IRQ_INDEX, channel);
        dma_channel_set_transfer_count(channel, current_task->encoding, false);
        dma_channel_set_read_addr(channel, current_task->originAddr, false);
        dma_channel_set_write_addr(channel, current_task->targetAddr, false);
        dma_channel_set_config(channel, &current_task->config, false);
        return true;
    }

    bool Device::start() const {
        corecheck(current_task != nullptr, RuntimeError("No current task for channel"));
        dma_channel_start(channel);
        return true;
    }

    bool Device::process(Transfer::Ptr task) { return configure(std::move(task)) && start(); }

    bool Device::restart() const {
        corecheck(current_task != nullptr, RuntimeError("No current task for channel"));

        this->kill();

        irq_counts[channel].store(0);
        dma_irqn_acknowledge_channel(IRQ_INDEX, channel);
        dma_channel_set_transfer_count(channel, current_task->encoding, false);
        dma_channel_set_read_addr(channel, current_task->originAddr, false);
        dma_channel_set_write_addr(channel, current_task->targetAddr, false);
        dma_channel_set_config(channel, &current_task->config, false);
        dma_channel_start(channel);

        return true;
    }

    void Device::setChannelIRQ(Handle handle) {
        IsrDaemon::get()->configure(channel, std::move(handle));
        irq_trigger.release();
    }
}; // namespace corekit::Dma
