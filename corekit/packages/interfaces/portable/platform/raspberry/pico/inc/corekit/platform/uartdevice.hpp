#pragma once
#include <hardware/uart.h>

#include <memory>
#include <span>

#include "corekit/platform/asyncdevice.hpp"

namespace corekit::platform {

    class UartDevice : public AsyncDevice<uint8_t> {
       public:
        using Ptr = std::shared_ptr<UartDevice>;

        UartDevice(                                               //
            uart_inst_t* instance = uart_default,                 //
            uint         baudRate = PICO_DEFAULT_UART_BAUD_RATE,  //
            uint         txPin    = PICO_DEFAULT_UART_TX_PIN,     //
            uint         rxPin    = PICO_DEFAULT_UART_RX_PIN      //
        );

        virtual bool write(const uint8_t& data) override;
        virtual bool write_bulk(std::span<const uint8_t> data) override;

        virtual bool read(uint8_t& data) override;
        virtual bool read_bulk(std::span<uint8_t> data) override;

       protected:
        virtual bool on_load() override;
        virtual bool on_unload() override;

        uart_inst_t* instance;
        uint         baudRate;
        uint         txPin;
        uint         rxPin;
    };

};  // namespace corekit::platform
