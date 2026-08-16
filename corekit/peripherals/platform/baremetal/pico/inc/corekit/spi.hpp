#pragma once

#include <hardware/spi.h>

#include "corekit/async.hpp"

namespace corekit {

    class SPI : public AsyncDevice<uint8_t> {
       public:
        using Ptr = std::shared_ptr<SPI>;

        SPI(spi_inst_t*        instance   = spi_default,               //
            bool               asyncRead  = false,                     //
            bool               asyncWrite = false,                     //
            const bool         slave      = true,                      //
            const float        freq       = 1e6,                       //
            const spi_cpol_t&  cpol       = SPI_CPOL_0,                //
            const spi_cpha_t&  cpha       = SPI_CPHA_1,                //
            const spi_order_t& order      = SPI_MSB_FIRST,             //
            const uint         txPin      = PICO_DEFAULT_SPI_TX_PIN,   //
            const uint         rxPin      = PICO_DEFAULT_SPI_RX_PIN,   //
            const uint         sckPin     = PICO_DEFAULT_SPI_SCK_PIN,  //
            const uint         csnPin     = PICO_DEFAULT_SPI_CSN_PIN   //
        );

        void loopback(bool enabled);

        virtual bool write(const uint8_t& data) override;
        virtual bool writeBulk(std::span<const uint8_t> data) override;
        virtual bool read(uint8_t& data) override;
        virtual bool readBulk(std::span<uint8_t> data) override;
        virtual bool xfer(const uint8_t& txData, uint8_t& rxData) override;
        virtual bool xferBulk(std::span<const uint8_t> txData,
                              std::span<uint8_t>       rxData) override;

       protected:
        virtual bool onLoad() override;
        virtual bool onUnload() override;

        spi_inst_t* instance;
        spi_cpol_t  cpol;
        spi_cpha_t  cpha;
        spi_order_t order;
        float       freq;
        bool        slave;
        uint        txPin;
        uint        rxPin;
        uint        sckPin;
        uint        csnPin;
    };

}  // namespace corekit