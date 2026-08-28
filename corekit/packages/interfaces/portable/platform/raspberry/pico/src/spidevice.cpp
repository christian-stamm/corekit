#include "corekit/platform/spidevice.hpp"

#include <hardware/gpio.h>
#include <hardware/spi.h>

#include <format>

namespace corekit::platform {

    SpiDevice::SpiDevice(spi_inst_t*        instance,  //
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
              std::format("SpiDevice{}", spi_get_index(instance)),        //
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

    bool SpiDevice::onLoad() {
        gpio_init(txPin);
        gpio_init(rxPin);
        gpio_init(sckPin);
        gpio_init(csnPin);

        gpio_set_dir(txPin, slave ? false : true);
        gpio_set_dir(rxPin, slave ? true : false);
        gpio_set_dir(sckPin, slave ? false : true);
        gpio_set_dir(csnPin, slave ? false : true);

        gpio_set_function(txPin, GPIO_FUNC_SPI);
        gpio_set_function(rxPin, GPIO_FUNC_SPI);
        gpio_set_function(sckPin, GPIO_FUNC_SPI);
        gpio_set_function(csnPin, GPIO_FUNC_SPI);

        spi_init(instance, freq);
        spi_set_slave(instance, slave);
        spi_set_format(instance, 8, cpol, cpha, order);

        return true;
    }

    bool SpiDevice::onUnload() {
        spi_deinit(instance);
        gpio_set_function(txPin, GPIO_FUNC_SIO);
        gpio_set_function(rxPin, GPIO_FUNC_SIO);
        gpio_set_function(sckPin, GPIO_FUNC_SIO);
        gpio_set_function(csnPin, GPIO_FUNC_SIO);
        return true;
    }

    void SpiDevice::loopback(bool enabled) {
        auto fn = enabled ? hw_set_bits : hw_clear_bits;
        fn(&spi_get_hw(instance)->cr1, SPI_SSPCR1_LBM_BITS);
    }

    bool SpiDevice::write(const uint8_t& data) {
        spi_write_blocking(instance, &data, 1);
        return true;
    }

    bool SpiDevice::writeBulk(std::span<const uint8_t> data) {
        return spi_write_blocking(instance, data.data(), data.size()) ==
               data.size();
    }

    bool SpiDevice::read(uint8_t& data) {
        return spi_read_blocking(instance, 0, &data, 1) == 1;
    }

    bool SpiDevice::readBulk(std::span<uint8_t> data) {
        return spi_read_blocking(instance, 0, data.data(), data.size()) ==
               data.size();
    }

    bool SpiDevice::xfer(const uint8_t& txData, uint8_t& rxData) {
        return spi_write_read_blocking(instance, &txData, &rxData, 1) == 1;
    }

    bool SpiDevice::xferBulk(std::span<const uint8_t> txData,
                             std::span<uint8_t>       rxData) {
        const uint num_cycles = std::min(txData.size(), rxData.size());

        return spi_write_read_blocking(instance,
                                       txData.data(),
                                       rxData.data(),
                                       num_cycles) == num_cycles;
    }

}  // namespace corekit::platform