#include "corekit/spidevice.hpp"

#include <hardware/spi.h>

#include <format>

#include "corekit/gpiodevice.hpp"

namespace corekit::Spi {

    Device::Device(spi_inst_t*        instance,  //
                   const bool         slave,     //
                   const float        freq,      //
                   const spi_cpol_t&  cpol,      //
                   const spi_cpha_t&  cpha,      //
                   const spi_order_t& order,     //
                   const uint         txPin,     //
                   const uint         rxPin,     //
                   const uint         sckPin,    //
                   const uint         csnPin     //
                   )
        : AsyncDevice(
              std::format("Device{}", spi_get_index(instance)),           //
              {&spi_get_hw(instance)->dr, spi_get_dreq(instance, true)},  //
              {&spi_get_hw(instance)->dr, spi_get_dreq(instance, false)}  //
              )
        , instance(instance)
        , slave(slave)
        , freq(freq)
        , cpol(cpol)
        , cpha(cpha)
        , order(order)
        , txPin(txPin)
        , rxPin(rxPin)
        , sckPin(sckPin)
        , csnPin(csnPin) {}

    bool Device::on_load() {
        Gpio::configure(txPin,
                        false,
                        false,
                        slave ? GPIO_IN : GPIO_OUT,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_SPI);
        Gpio::configure(rxPin,
                        false,
                        false,
                        slave ? GPIO_OUT : GPIO_IN,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_SPI);
        Gpio::configure(sckPin,
                        false,
                        false,
                        slave ? GPIO_IN : GPIO_OUT,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_SPI);
        Gpio::configure(csnPin,
                        false,
                        false,
                        slave ? GPIO_IN : GPIO_OUT,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_SPI);

        spi_init(instance, freq);
        spi_set_slave(instance, slave);
        spi_set_format(instance, 8, cpol, cpha, order);

        return true;
    }

    bool Device::on_unload() {
        spi_deinit(instance);
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
        Gpio::configure(sckPin,
                        false,
                        false,
                        GPIO_IN,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_SIO);
        Gpio::configure(csnPin,
                        false,
                        false,
                        GPIO_IN,
                        GPIO_OVERRIDE_NORMAL,
                        GPIO_FUNC_SIO);
        return true;
    }

    void Device::loopback(bool enabled) {
        auto fn = enabled ? hw_set_bits : hw_clear_bits;
        fn(&spi_get_hw(instance)->cr1, SPI_SSPCR1_LBM_BITS);
    }

    bool Device::write(const uint8_t& data) {
        spi_write_blocking(instance, &data, 1);
        return true;
    }

    bool Device::write_bulk(std::span<const uint8_t> data) {
        return spi_write_blocking(instance, data.data(), data.size()) ==
               data.size();
    }

    bool Device::read(uint8_t& data) {
        return spi_read_blocking(instance, 0, &data, 1) == 1;
    }

    bool Device::read_bulk(std::span<uint8_t> data) {
        return spi_read_blocking(instance, 0, data.data(), data.size()) ==
               data.size();
    }

    bool Device::xfer(const uint8_t& txData, uint8_t& rxData) {
        return spi_write_read_blocking(instance, &txData, &rxData, 1) == 1;
    }

    bool Device::xferBulk(std::span<const uint8_t> txData,
                          std::span<uint8_t>       rxData) {
        const uint num_cycles = std::min(txData.size(), rxData.size());

        return spi_write_read_blocking(instance,
                                       txData.data(),
                                       rxData.data(),
                                       num_cycles) == num_cycles;
    }

}  // namespace corekit::Spi