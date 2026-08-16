#include "corekit/uart.hpp"

#include <hardware/gpio.h>
#include <hardware/uart.h>

#include <string>

namespace corekit {

    UART::UART(                 //
        uart_inst_t* instance,  //
        uint         baudRate,  //
        uint         txPin,     //
        uint         rxPin      //
        )
        : AsyncDevice(
              "UART" + std::to_string(uart_get_index(instance)),
              {&uart_get_hw(instance)->dr, uart_get_dreq(instance, true)},  //
              {&uart_get_hw(instance)->dr, uart_get_dreq(instance, false)}  //
              )
        , instance(instance)
        , baudRate(baudRate)
        , txPin(txPin)
        , rxPin(rxPin) {}

    bool UART::onLoad() {
        uart_init(instance, baudRate);
        gpio_set_function(txPin, GPIO_FUNC_UART);
        gpio_set_function(rxPin, GPIO_FUNC_UART);
        return true;
    }

    bool UART::onUnload() {
        uart_deinit(instance);
        gpio_set_function(txPin, GPIO_FUNC_SIO);
        gpio_set_function(rxPin, GPIO_FUNC_SIO);
        return true;
    }

    bool UART::write(const uint8_t& data) {
        uart_putc(instance, static_cast<char>(data));
        return true;
    }

    bool UART::writeBulk(std::span<const uint8_t> data) {
        uart_write_blocking(instance,
                            data.data(),
                            static_cast<size_t>(data.size()));
        return true;
    }

    bool UART::read(uint8_t& data) {
        int c = uart_getc(instance);
        if (c == PICO_ERROR_TIMEOUT) {
            return false;
        }
        data = static_cast<uint8_t>(c);
        return true;
    }

    bool UART::readBulk(std::span<uint8_t> data) {
        uart_read_blocking(instance,
                           data.data(),
                           static_cast<size_t>(data.size()));

        return true;
    }

}  // namespace corekit
