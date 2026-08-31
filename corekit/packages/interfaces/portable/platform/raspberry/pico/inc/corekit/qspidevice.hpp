#pragma once

#include <hardware/pio.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

#include "corekit/atomic.hpp"
#include "corekit/basedevice.hpp"
#include "corekit/dmadevice.hpp"
#include "corekit/gpiodevice.hpp"
#include "corekit/mutex.hpp"
#include "corekit/piodevice.hpp"
#include "corekit/semaphore.hpp"

namespace corekit::Qspi {

    /** Native RP2 PIO-backed quad-SPI transport.
 *
 * The transport is intentionally protocol agnostic: command/address framing is
 * supplied by the caller as a byte prefix. Both the prefix and payload are
 * shifted over four data lines. Protocols that serialize an opcode on IO0
 * (such as W6300 QSPI) can pre-expand that opcode in the prefix.
 *
 * Transfers are DMA paced. write() completion is sourced by a token emitted by
 * the PIO state machine after the final nibble reached the pins; no TXSTALL
 * polling or Pico SDK blocking DMA helper is used.
 *
 * begin_*() is nonblocking. All spans must remain valid until the completion
 * callback runs. write()/read() are cooperative wrappers that suspend on a
 * CoreKit semaphore rather than busy-waiting.
 */
    struct Config {
            PIO         pio{pio0};
            Gpio::Range data{19, 4}; // IO0..IO3 must be consecutive
            Gpio::Pin   clock_pin{18};

            // Requested SCK. The PIO program uses two instructions per clock period;
            // the driver derives the divider from clk_sys and clamps it to the PIO
            // hardware minimum. On a 150 MHz RP2350, 75 MHz maps to divider 1.0.
            uint32_t clock_hz{75'000'000};

            gpio_drive_strength drive_strength{GPIO_DRIVE_STRENGTH_12MA};
            gpio_slew_rate      slew_rate{GPIO_SLEW_RATE_FAST};
            bool                pull_down{true};
            bool                bypass_input_sync{true};
    };

    class Device final : public BaseDevice {
        public:

            using Ptr        = std::shared_ptr<Device>;
            using Completion = std::function<void(bool)>;

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
            bool busy() const noexcept {
                return busy_.load();
            }

            [[nodiscard]]
            uint32_t actual_clock_hz() const noexcept;

            bool begin_write(std::span<const uint8_t> prefix, std::span<const uint8_t> payload, Completion completion = {});
            bool begin_read(std::span<const uint8_t> prefix, std::span<uint8_t> payload, Completion completion = {});

            bool write(std::span<const uint8_t> prefix, std::span<const uint8_t> payload = {});
            bool read(std::span<const uint8_t> prefix, std::span<uint8_t> payload);

            void cancel();

        protected:

            bool on_load() override;
            bool on_unload() override;

        private:

            class Engine;

            enum class Operation : uint8_t { None, Read, Write };

            bool validate_config() const;
            bool prepare_write(std::size_t byte_count);
            bool prepare_read(std::size_t prefix_bytes, std::size_t payload_bytes);
            bool configure_tx(Dma::Device& dma, std::span<const uint8_t> source, int chain_to = -1);
            bool configure_rx(std::span<uint8_t> target);
            bool configure_write_completion();
            void complete(bool ok);
            void quiesce();

            Config config_;

            std::shared_ptr<Pio::Program> program_;
            std::shared_ptr<Engine>       engine_;
            Dma::Device::Ptr              prefix_dma_;
            Dma::Device::Ptr              payload_dma_;
            Dma::Device::Ptr              rx_dma_;

            mutable Mutex state_mutex_;
            Atomic<bool>  busy_{false};
            Operation     operation_{Operation::None};
            Completion    completion_{};
            uint32_t      write_completion_token_{0};
    };

} // namespace corekit::Qspi
