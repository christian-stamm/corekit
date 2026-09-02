#pragma once

#include <hardware/dma.h>
#include <hardware/irq.h>
#include <pico/types.h>

#include <cstdint>
#include <functional>
#include <memory>

#include "corekit/asyncdevice.hpp"

namespace corekit::Dma {

    using Config   = dma_channel_config;
    using Handle   = std::function<void()>;
    using XferSize = dma_channel_transfer_size;
    using AddrUpdt = dma_address_update_type_t;

    enum class Wrapping { None = 0, Read = 1, Write = 2 };

    struct Transfer {
        friend class Device;

       public:
        using Ptr = std::shared_ptr<Transfer>;

        Transfer(uint                 channel,
                 uint                 dreq,
                 const volatile void* originAddr,
                 const volatile void* targetAddr,
                 AddrUpdt             originUpdate,
                 AddrUpdt             targetUpdate,
                 uint32_t             length,
                 XferSize             blockSize,
                 Wrapping             wrapping = Wrapping::None,
                 bool                 byteswap = false,
                 bool                 sniff    = false,
                 int                  chain    = -1,
                 Handle&&             handle   = nullptr);

        ~Transfer();

        template <typename T>
        static Ptr mem2dev(uint               chn,
                           std::span<const T> src,
                           const CtrlBlock&   dst,
                           bool               repeat   = false,
                           bool               reverse  = false,
                           bool               byteswap = false,
                           bool               sniff    = false,
                           int                chain    = -1,
                           Handle&&           handle   = nullptr);

        template <typename T>
        static Ptr dev2mem(uint             chn,
                           const CtrlBlock& src,
                           std::span<T>     dst,
                           bool             repeat   = false,
                           bool             reverse  = false,
                           bool             byteswap = false,
                           bool             sniff    = false,
                           int              chain    = -1,
                           Handle&&         handle   = nullptr);

        template <typename T>
        static Ptr mem2mem(uint               chn,
                           std::span<const T> src,
                           std::span<T>       dst,
                           bool               reverse  = false,
                           bool               byteswap = false,
                           Handle&&           handle   = nullptr);

       protected:
        volatile void* originAddr;
        volatile void* targetAddr;
        uint32_t       encoding;
        uint           channel;
        Config         config;
    };

    class Device : public AsyncDevice<uint32_t> {
       public:
        using Ptr = std::shared_ptr<Device>;

        Device(uint channel);
        virtual ~Device() override;

        static Ptr requestUnused();

        static void enableIRQ();
        static void disableIRQ();

        bool process(Transfer::Ptr task);
        void chaining(uint channel);
        void sniffing(bool enabled);

        bool busy() const;
        void kill() const;

        const uint channel;

        virtual bool write(const uint32_t& value) override {
            return false;
        }

        virtual bool read(uint32_t& value) override {
            return false;
        }

       protected:
        virtual bool on_load() override;
        virtual bool on_unload() override;

       private:
        void setIRQ(Handle handle = nullptr, bool quiet = false);
    };

}  // namespace corekit::Dma
