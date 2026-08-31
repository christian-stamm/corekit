#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "corekit/basedevice.hpp"
#include "corekit/mutex.hpp"
#include "corekit/qspidevice.hpp"

namespace corekit::Eth {

    struct Ipv4Address {
            std::array<uint8_t, 4> octets{};
    };

    struct Endpoint {
            Ipv4Address address{};
            uint16_t    port{0};
    };

    struct NetworkConfig {
            std::array<uint8_t, 6> mac{0x02, 0x00, 0x00, 0x00, 0x00, 0x01};
            Ipv4Address            address{
                           {192, 168, 1, 177}
            };
            Ipv4Address subnet{
                {255, 255, 255, 0}
            };
            Ipv4Address gateway{
                {192, 168, 1, 1}
            };
            Ipv4Address dns{
                {0, 0, 0, 0}
            };
    };

    struct Config {
            // W6300 QSPI uses four consecutive data GPIOs. CoreKit's QSPI transport is
            // PIO-backed and the physical SCK rate is clk_sys / (2 * clock_divider).
            Qspi::Config bus{};

            uint cs_pin{17};
            uint reset_pin{16};
            uint irq_pin{15};

            uint8_t       socket{0};
            uint16_t      local_port{5000};
            NetworkConfig network{};
    };

    /** W6300 UDP device for Raspberry Pi Pico / RP2.
 *
 * WIZnet's ioLibrary provides register and socket semantics only. CoreKit owns
 * every platform primitive: PIO QSPI, DMA, GPIO, mutexes, semaphores and time.
 * No WIZnet Pico port source is linked.
 *
 * UDP is configured nonblocking. send_to()/receive_from() therefore return 0
 * when the socket would block. QSPI register transactions suspend on CoreKit
 * synchronization primitives while DMA/PIO execute; they never busy-spin.
 */
    class Device final : public BaseDevice {
        public:

            using Ptr = std::shared_ptr<Device>;

            explicit Device(Config config = {});
            ~Device() override;

            Device(const Device&)            = delete;
            Device(Device&&)                 = delete;
            Device& operator=(const Device&) = delete;
            Device& operator=(Device&&)      = delete;

            [[nodiscard]]
            const Config& config() const noexcept {
                return config_;
            }

            [[nodiscard]]
            bool link_up() const;

            int32_t send_to(std::span<const uint8_t> payload, const Endpoint& remote);
            int32_t receive_from(std::span<uint8_t> payload, Endpoint& remote);

        protected:

            bool on_load() override;
            bool on_unload() override;

        private:

            bool init_transport();
            bool init_chip();
            bool open_udp_socket();
            void reset_chip();
            void chip_select(bool selected);

            bool qspi_write_frame(uint8_t opcode, uint16_t address, std::span<const uint8_t> payload);
            bool qspi_read_frame(uint8_t opcode, uint16_t address, std::span<uint8_t> payload);

            static Device* active_;
            static void    critical_enter_cb();
            static void    critical_exit_cb();
            static void    cs_select_cb();
            static void    cs_deselect_cb();
            static void    qspi_read_cb(uint8_t opcode, uint16_t address, uint8_t* data, uint16_t length);
            static void    qspi_write_cb(uint8_t opcode, uint16_t address, uint8_t* data, uint16_t length);

            Config        config_;
            Qspi::Device  qspi_;
            mutable Mutex transaction_mutex_;
    };

} // namespace corekit::Eth
