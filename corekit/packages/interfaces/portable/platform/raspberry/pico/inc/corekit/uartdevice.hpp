#pragma once
#include <hardware/uart.h>

#include <memory>
#include <span>

#include "corekit/asyncdevice.hpp"

namespace corekit::Uart {

    class Device : public AsyncDevice<uint8_t> {
       public:
        using Ptr = std::shared_ptr<Device>;

        Device(                                                   //
            uart_inst_t* instance = uart_default,                 //
            uint         baudRate = PICO_DEFAULT_UART_BAUD_RATE,  //
            uint         txPin    = PICO_DEFAULT_UART_TX_PIN,     //
            uint         rxPin    = PICO_DEFAULT_UART_RX_PIN      //
        );

        virtual VoidResult write(const uint8_t& data) override;
        virtual VoidResult write_burst(std::span<const uint8_t> data) override;

        virtual VoidResult read(uint8_t& data) override;
        virtual VoidResult read_burst(std::span<uint8_t> data) override;

       protected:
        virtual VoidResult on_load() override;
        virtual VoidResult on_unload() override;

        uart_inst_t* instance;
        uint         baudRate;
        uint         txPin;
        uint         rxPin;
    };

};  // namespace corekit::Uart
