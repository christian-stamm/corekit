#pragma once

#include <hardware/dma.h>
#include <hardware/irq.h>
#include <pico/types.h>

#include <cstdint>
#include <functional>
#include <memory>

#include "corekit/platform/asyncdevice.hpp"

template <typename T>
class DmaDevice : public corekit::AsyncDevice<T> {
   public:
    using Ptr      = std::shared_ptr<DmaDevice>;
    using Config   = dma_channel_config;
    using Handle   = std::function<void()>;
    using XferSize = dma_channel_transfer_size;
    using AddrUpdt = dma_address_update_type_t;

    enum Mode { DEV2DEV, MEM2DEV, DEV2MEM, MEM2MEM };

    struct Transfer {
        CtrlBlock origin;
        CtrlBlock target;
        uint32_t  length;
    };

    DMA(uint channel, bool highPrio = false);
    virtual ~DMA() override;

    static Ptr requestUnused();

    static void enableIRQ();
    static void disableIRQ();

    template <typename T>
    void configure(const Transfer& task, bool verify = true);

    template <typename T>
    void process(const Transfer& task,
                 bool            configure,
                 bool            wait = false,
                 bool            kill = false);

    void setMode(Mode mode     = MEM2MEM,
                 uint dreq     = DREQ_FORCE,
                 bool reversed = false,
                 bool byteswap = false);

    void setIRQ(Handle handle = nullptr, bool quiet = false);
    void wrapping(bool wrapWrite = false, uint wrapBytes = 0);
    void chaining(uint channel);
    void sniffing(bool enabled);
    bool busy() const;
    void kill() const;
    void block() const;

    const uint channel;

   protected:
    virtual bool onLoad() override;
    virtual bool onUnload() override;

   private:
    Config   config;
    uint     wrapBytes;
    bool     wrapWrite;
    uint32_t encoding;
};