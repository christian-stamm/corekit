#include "corekit/uartdevice.hpp"

#include <hardware/uart.h>

#include <format>

#include "corekit/gpiodevice.hpp"
#include "corekit/result.hpp"

namespace corekit::Uart {

    Device::Device(             //
        uart_inst_t* instance,  //
        uint         baudRate,  //
        uint         txPin,     //
        uint         rxPin      //
        )
        : AsyncDevice(
              std::format("UART{}", uart_get_index(instance)),              //
              {&uart_get_hw(instance)->dr, uart_get_dreq(instance, true)},  //
              {&uart_get_hw(instance)->dr, uart_get_dreq(instance, false)}  //
              )
        , instance(instance)
        , baudRate(baudRate)
        , txPin(txPin)
        , rxPin(rxPin) {}

    VoidResult Device::on_load() {
        uart_init(instance, baudRate);
        Gpio::configure(txPin,
                        false,
                        false,
                        GPIO_OUT,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_UART);
        Gpio::configure(rxPin,
                        false,
                        false,
                        GPIO_IN,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_UART);
        return VoidResult();
    }

    VoidResult Device::on_unload() {
        uart_deinit(instance);
        Gpio::configure(txPin,
                        false,
                        false,
                        GPIO_IN,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_SIO);
        Gpio::configure(rxPin,
                        false,
                        false,
                        GPIO_IN,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_SIO);
        return VoidResult();
    }

    VoidResult Device::write(const uint8_t& data) {
        uart_putc(instance, static_cast<char>(data));
        return VoidResult();
    }

    VoidResult Device::write_burst(std::span<const uint8_t> data) {
        uart_write_blocking(instance,
                            data.data(),
                            static_cast<size_t>(data.size()));
        return VoidResult();
    }

    VoidResult Device::read(uint8_t& data) {
        int c = uart_getc(instance);
        if (c == PICO_ERROR_TIMEOUT) {
            return RuntimeError("Failed to read data.");
        }
        data = static_cast<uint8_t>(c);
        return VoidResult();
    }

    VoidResult Device::read_burst(std::span<uint8_t> data) {
        uart_read_blocking(instance,
                           data.data(),
                           static_cast<size_t>(data.size()));

        return VoidResult();
    }

}  // namespace corekit::Uart