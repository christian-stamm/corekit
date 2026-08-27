#include "corekit/platform/uartdevice.hpp"

#include <hardware/gpio.h>
#include <hardware/uart.h>

#include <format>

namespace corekit::platform {

    UartDevice::UartDevice(     //
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

    bool UartDevice::on_load() {
        uart_init(instance, baudRate);
        gpio_set_function(txPin, GPIO_FUNC_UART);
        gpio_set_function(rxPin, GPIO_FUNC_UART);
        return true;
    }

    bool UartDevice::on_unload() {
        uart_deinit(instance);
        gpio_set_function(txPin, GPIO_FUNC_SIO);
        gpio_set_function(rxPin, GPIO_FUNC_SIO);
        return true;
    }

    bool UartDevice::write(const uint8_t& data) {
        uart_putc(instance, static_cast<char>(data));
        return true;
    }

    bool UartDevice::write_bulk(std::span<const uint8_t> data) {
        uart_write_blocking(instance,
                            data.data(),
                            static_cast<size_t>(data.size()));
        return true;
    }

    bool UartDevice::read(uint8_t& data) {
        int c = uart_getc(instance);
        if (c == PICO_ERROR_TIMEOUT) {
            return false;
        }
        data = static_cast<uint8_t>(c);
        return true;
    }

    bool UartDevice::read_bulk(std::span<uint8_t> data) {
        uart_read_blocking(instance,
                           data.data(),
                           static_cast<size_t>(data.size()));

        return true;
    }

}  // namespace corekit::platform