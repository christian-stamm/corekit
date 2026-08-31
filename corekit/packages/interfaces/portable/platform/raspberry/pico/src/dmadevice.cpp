#include <hardware/dma.h>
#include <hardware/platform_defs.h>
#include <hardware/regs/dreq.h>
#include <pico/time.h>
#include <pico/types.h>

#include <cmath>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <stdexcept>

#include "device/device.hpp"
#include "device/dma.hpp"
#include "math.hpp"

constexpr uint IRQ_INDEX = 0;
constexpr uint IRQ_NUM   = DMA_IRQ_NUM(IRQ_INDEX);

constexpr uint MIN_RING_BITS = 1;
constexpr uint MAX_RING_BITS = 15;

__isr void event();

std::array<DMA::Handle, NUM_DMA_CHANNELS> handles;

DMA::DMA(uint channel, bool highPrio)
    : Device<uint32_t>(
          std::format("DMA{}", channel),                                     //
          {&dma_channel_hw_addr(channel)->al2_write_addr_trig, DREQ_FORCE},  //
          {&dma_channel_hw_addr(channel)->al3_read_addr_trig, DREQ_FORCE}    //
          )
    , channel(channel)
    , config(dma_channel_get_default_config(channel))
    , wrapBytes(0)
    , wrapWrite(false) {
    dma_channel_claim(channel);
    channel_config_set_high_priority(&config, highPrio);
}

DMA::~DMA() {
    dma_channel_unclaim(channel);
}

void DMA::prepare() {
    // Nothing to do
}

void DMA::cleanup() {
    kill();
    dma_channel_cleanup(channel);
}

DMA::Ptr DMA::requestUnused() {
    for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
        if (!dma_channel_is_claimed(channel)) {
            return std::make_shared<DMA>(channel);
        }
    }

    throw std::runtime_error("No unused Dma channels available.");
}

bool DMA::busy() const {
    return dma_channel_is_busy(channel);
}

void DMA::kill() const {
    if (busy()) {
        dma_channel_abort(channel);
    }
}

void DMA::block() const {
    dma_channel_wait_for_finish_blocking(channel);
}

void DMA::setMode(Mode mode, uint dreq, bool reversed, bool byteswap) {
    AddrUpdt writeUpdate = DMA_ADDRESS_UPDATE_NONE;
    AddrUpdt readUpdate  = DMA_ADDRESS_UPDATE_NONE;

    if (mode == MEM2DEV || mode == MEM2MEM) {
        readUpdate = reversed ? DMA_ADDRESS_UPDATE_DECREMENT
                              : DMA_ADDRESS_UPDATE_INCREMENT;
    }

    if (mode == DEV2MEM || mode == MEM2MEM) {
        writeUpdate = reversed ? DMA_ADDRESS_UPDATE_DECREMENT
                               : DMA_ADDRESS_UPDATE_INCREMENT;
    }

    channel_config_set_dreq(&config, dreq);
    channel_config_set_bswap(&config, byteswap);
    channel_config_set_read_address_update_type(&config, readUpdate);
    channel_config_set_write_address_update_type(&config, writeUpdate);
}

void DMA::setIRQ(Handle handle, bool quiet) {
    handles[channel] = handle;
    channel_config_set_irq_quiet(&config, quiet);
    dma_irqn_set_channel_enabled(IRQ_INDEX, channel, handle != nullptr);
}

void DMA::wrapping(bool wrapWrite, uint wrapBytes) {
    size_t ringSize = 0;

    if (0 < wrapBytes) {
        ringSize = std::log2(wrapBytes);

        const bool overflow  = MAX_RING_BITS < ringSize;
        const bool underflow = MIN_RING_BITS > ringSize;

        if (overflow || underflow || !math::isPow2(wrapBytes)) {
            throw std::runtime_error(
                "Ringbuffer size must be a power of two between 2^1 and 2^15.");
        }
    }

    this->wrapBytes = wrapBytes;
    this->wrapWrite = wrapWrite;
    channel_config_set_ring(&config, wrapWrite, ringSize);
}

void DMA::chaining(uint channel) {
    channel_config_set_chain_to(&config, channel);
}

void DMA::sniffing(bool enabled) {
    channel_config_set_sniff_enable(&config, enabled);
}

template <typename T>
void DMA::configure(const Transfer& task, bool verify) {
    const XferSize blockSize = static_cast<XferSize>(std::log2(sizeof(T)));
    const bool     doWrap    = 0 < wrapBytes;

    if (verify) {
        if (task.length == 0) {
            throw std::runtime_error(
                "Transfer length must be greater than zero.");
        }

        if (!task.origin || !task.target) {
            throw std::runtime_error(
                "Transfer source and destination must be specified.");
        }

        if (task.repeat && !doWrap) {
            throw std::runtime_error(
                "Cannot use repeating transfers without ringbuffers "
                "(Wrapping).");
        }
    }

    encoding = dma_encode_transfer_count(task.length);

    if (doWrap) {
        const bool isAligned =
            Memory::isAligned(wrapWrite ? task.target : task.origin, wrapBytes);

        if (!isAligned) {
            throw std::runtime_error("Ringbuffer address is not aligned.");
        }

        if (task.repeat) {
            encoding = dma_encode_transfer_count_with_self_trigger(encoding);
        }
    } else if (task.repeat) {
        throw std::runtime_error(
            "Cannot use repeating transfers without ringbuffers.");
    }

    channel_config_set_transfer_data_size(&config, blockSize);
    channel_config_set_enable(&config, true);

    dma_channel_set_read_addr(channel,
                              const_cast<volatile void*>(task.origin),
                              false);
    dma_channel_set_write_addr(channel,
                               const_cast<volatile void*>(task.target),
                               false);
    dma_channel_set_transfer_count(channel, encoding, false);
    dma_channel_set_config(channel, &config, false);
}

template <typename T>
void DMA::process(const Transfer& task,
                  bool            doConfigure,
                  bool            doWait,
                  bool            doKill) {
    if (doKill) {
        this->kill();
    }

    if (busy()) {
        throw std::runtime_error(
            "Cannot start a new transfer on a busy channel.");
    }

    if (doConfigure) {
        this->configure<T>(task);
    }

    dma_channel_start(channel);

    if (doWait) {
        this->block();
    }
}

void DMA::enableIRQ() {
    irq_set_exclusive_handler(IRQ_NUM, event);
    irq_set_enabled(IRQ_NUM, true);
}

void DMA::disableIRQ() {
    irq_set_enabled(IRQ_NUM, false);
    irq_remove_handler(IRQ_NUM, event);
}

void event() {
    for (uint channel = 0; channel < NUM_DMA_CHANNELS; channel++) {
        if (dma_irqn_get_channel_status(IRQ_INDEX, channel)) {
            const DMA::Handle handle = handles[channel];

            if (handle) {
                handle();
            }

            dma_irqn_acknowledge_channel(IRQ_INDEX, channel);
        }
    }
}

template void DMA::process<uint8_t>(const Transfer&, bool, bool, bool);
template void DMA::process<uint16_t>(const Transfer&, bool, bool, bool);
template void DMA::process<uint32_t>(const Transfer&, bool, bool, bool);