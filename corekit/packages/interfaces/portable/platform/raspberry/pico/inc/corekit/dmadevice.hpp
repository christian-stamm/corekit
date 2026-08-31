#pragma once

#include <hardware/dma.h>
#include <hardware/irq.h>
#include <pico/types.h>

#include <cstdint>
#include <functional>
#include <memory>

#include "corekit/asyncdevice.hpp"
#include "corekit/task.hpp"

namespace corekit::Dma {

    using Config   = dma_channel_config;
    using Handle   = std::function<void(int)>;
    using XferSize = dma_channel_transfer_size;
    using AddrUpdt = dma_address_update_type_t;

    struct Transfer {
            friend class Device;

        public:

            using Ptr  = std::shared_ptr<Transfer>;
            using List = std::vector<Ptr>;

            Transfer(
                uint                 channel,
                const volatile void* originAddr = nullptr,
                AddrUpdt originUpdate           = AddrUpdt::DMA_ADDRESS_UPDATE_NONE,
                const volatile void* targetAddr = nullptr,
                AddrUpdt targetUpdate           = AddrUpdt::DMA_ADDRESS_UPDATE_NONE,
                uint32_t burst_length           = 0,
                int32_t wrap_length             = 0,
                // wrap_length < 0 for read // 0 < wrap_length for write
                XferSize blockSize = Dma::XferSize::DMA_SIZE_32,
                uint dreq          = DREQ_FORCE,
                bool repeat        = false,
                bool byteswap      = false,
                bool sniff         = false,
                int chain          = -1);

        protected:

            volatile void* originAddr;
            volatile void* targetAddr;
            uint32_t       encoding;
            uint           channel;
            Config         config;
    };

    class Device : public AsyncDevice<uint32_t> {
        public:

            using Ptr  = std::shared_ptr<Device>;
            using List = std::vector<Ptr>;

            Device();
            ~Device();

            static Ptr request_unused();

            void setChannelIRQ(Handle handle = nullptr);

            bool configure(Transfer::Ptr task);
            bool start() const;
            bool process(Transfer::Ptr task);
            void chaining(uint channel);
            void sniffing(bool enabled);

            bool busy() const;
            void kill() const;

            bool restart() const;

            const uint channel;

            virtual bool write(const uint32_t& value) override { return false; }

            virtual bool read(uint32_t& value) override { return false; }

        protected:

            Device(uint channel);

            virtual bool on_load() override;
            virtual bool on_unload() override;

        private:

            void setIRQ(Handle handle = nullptr, bool quiet = false);

            Transfer::Ptr current_task;
    };

    class IsrDaemon : public Task {
        public:

            using Ptr = std::shared_ptr<IsrDaemon>;

            static Ptr get() {
                static Ptr daemon = Ptr(new IsrDaemon());
                return daemon;
            }

            bool configure(uint channel, Handle handle = nullptr);
            void spin_once() const;

        protected:

            virtual VoidResult on_init(StopToken token) override;
            virtual VoidResult on_exec(StopToken token) override;
            virtual VoidResult on_exit(StopToken token) override;

            void reconfigure();

        private:

            IsrDaemon();
    };

} // namespace corekit::Dma
