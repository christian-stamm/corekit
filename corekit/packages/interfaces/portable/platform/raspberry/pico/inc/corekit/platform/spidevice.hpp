#pragma once
#include <hardware/spi.h>

#include <memory>
#include <span>

#include "corekit/platform/asyncdevice.hpp"

namespace corekit::platform::spi {

    class Device : public AsyncDevice<uint8_t> {
       public:
        using Ptr = std::shared_ptr<Device>;

        Device(spi_inst_t*        instance = spi_default,               //
               const bool         slave    = true,                      //
               const float        freq     = 1e6,                       //
               const spi_cpol_t&  cpol     = SPI_CPOL_0,                //
               const spi_cpha_t&  cpha     = SPI_CPHA_1,                //
               const spi_order_t& order    = SPI_MSB_FIRST,             //
               const uint         txPin    = PICO_DEFAULT_SPI_TX_PIN,   //
               const uint         rxPin    = PICO_DEFAULT_SPI_RX_PIN,   //
               const uint         sckPin   = PICO_DEFAULT_SPI_SCK_PIN,  //
               const uint         csnPin   = PICO_DEFAULT_SPI_CSN_PIN   //
        );

        void loopback(bool enabled);

        virtual bool write(const uint8_t& data) override;
        virtual bool write_bulk(std::span<const uint8_t> data) override;

        virtual bool read(uint8_t& data) override;
        virtual bool read_bulk(std::span<uint8_t> data) override;

        virtual bool xfer(const uint8_t& txData, uint8_t& rxData) override;
        virtual bool xferBulk(std::span<const uint8_t> txData,
                              std::span<uint8_t>       rxData) override;

       protected:
        virtual bool on_load() override;
        virtual bool on_unload() override;

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

};  // namespace corekit::platform::spi
