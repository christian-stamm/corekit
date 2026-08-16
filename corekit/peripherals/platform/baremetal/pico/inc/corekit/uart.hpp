#pragma once

#include <hardware/uart.h>

#include "corekit/async.hpp"

namespace corekit {

    class UART : public AsyncDevice<uint8_t> {
       public:
        using Ptr = std::shared_ptr<UART>;

        UART(                                                     //
            uart_inst_t* instance = uart_default,                 //
            uint         baudRate = PICO_DEFAULT_UART_BAUD_RATE,  //
            uint         txPin    = PICO_DEFAULT_UART_TX_PIN,     //
            uint         rxPin    = PICO_DEFAULT_UART_RX_PIN      //
        );

        virtual bool write(const uint8_t& data) override;
        virtual bool writeBulk(std::span<const uint8_t> data) override;
        virtual bool read(uint8_t& data) override;
        virtual bool readBulk(std::span<uint8_t> data) override;

       protected:
        virtual bool onLoad() override;
        virtual bool onUnload() override;

        uart_inst_t* instance;
        uint         baudRate;
        uint         txPin;
        uint         rxPin;
    };

}  // namespace corekit