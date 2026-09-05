#include "corekit/spidevice.hpp"

#include <hardware/spi.h>

#include <format>

#include "corekit/error.hpp"
#include "corekit/gpiodevice.hpp"
#include "corekit/result.hpp"

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

    VoidResult Device::on_load() {
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

        return VoidResult();
    }

    VoidResult Device::on_unload() {
        spi_deinit(instance);

        if (!(Gpio::configure(txPin,
                              false,
                              false,
                              GPIO_IN,
                              GPIO_OVERRIDE_NORMAL,
                              GPIO_FUNC_SIO) &&

              Gpio::configure(rxPin,
                              false,
                              false,
                              GPIO_IN,
                              GPIO_OVERRIDE_NORMAL,
                              GPIO_FUNC_SIO) &&

              Gpio::configure(sckPin,
                              false,
                              false,
                              GPIO_IN,
                              GPIO_OVERRIDE_NORMAL,
                              GPIO_FUNC_SIO) &&

              Gpio::configure(csnPin,
                              false,
                              false,
                              GPIO_IN,
                              GPIO_OVERRIDE_NORMAL,
                              GPIO_FUNC_SIO))) {
            return RuntimeError("Failed to reset SPI pins to GPIO.");
        }

        return VoidResult();
    }

    void Device::loopback(bool enabled) {
        auto fn = enabled ? hw_set_bits : hw_clear_bits;
        fn(&spi_get_hw(instance)->cr1, SPI_SSPCR1_LBM_BITS);
    }

    VoidResult Device::write(const uint8_t& data) {
        spi_write_blocking(instance, &data, 1);
        return VoidResult();
    }

    VoidResult Device::write_burst(std::span<const uint8_t> data) {
        if (spi_write_blocking(instance, data.data(), data.size()) !=
            data.size()) {
            return RuntimeError("Failed to write burst data.");
        }

        return VoidResult();
    }

    VoidResult Device::read(uint8_t& data) {
        if (spi_read_blocking(instance, 0, &data, 1) != 1) {
            return RuntimeError("Failed to read data.");
        }

        return VoidResult();
    }

    VoidResult Device::read_burst(std::span<uint8_t> data) {
        if (spi_read_blocking(instance, 0, data.data(), data.size()) !=
            data.size()) {
            return RuntimeError("Failed to read burst data.");
        }

        return VoidResult();
    }

    VoidResult Device::xfer(const uint8_t& txData, uint8_t& rxData) {
        if (spi_write_read_blocking(instance, &txData, &rxData, 1) != 1) {
            return RuntimeError("Failed to xfer data.");
        }

        return VoidResult();
    }

    VoidResult Device::xfer_burst(std::span<const uint8_t> txData,
                                  std::span<uint8_t>       rxData) {
        const uint num_cycles = txData.size();

        if (rxData.size() < num_cycles) {
            return RuntimeError("rxData span is smaller than txData span.");
        }

        if (spi_write_read_blocking(instance,
                                    txData.data(),
                                    rxData.data(),
                                    num_cycles) != num_cycles) {
            return RuntimeError("Failed to xfer burst data.");
        }

        return VoidResult();
    }

}  // namespace corekit::Spi