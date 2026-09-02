#include "corekit/gpiodevice.hpp"

#include <hardware/gpio.h>
#include <hardware/irq.h>
#include <pico/types.h>

#include <format>
#include <map>

namespace corekit::Gpio {

    std::map<uint, Handle> callbacks;

    __isr inline void isrEvent(uint gpio, uint32_t events) {
        if (callbacks.contains(gpio)) {
            const auto handle = callbacks[gpio];

            if (handle) {
                handle(gpio, static_cast<gpio_irq_level>(events));
            }
        }

        gpio_acknowledge_irq(gpio, events);
    }

    void enableIRQ() {
        irq_set_enabled(IO_IRQ_BANK0, true);
    }

    void disableIRQ() {
        irq_set_enabled(IO_IRQ_BANK0, false);
    }

    void setHandle(uint pin, gpio_irq_level event, const Handle& callback) {
        callbacks[pin] = callback;
        gpio_set_irq_enabled_with_callback(pin,
                                           static_cast<uint32_t>(event),
                                           callback != nullptr,
                                           isrEvent);
    }

    void configure(uint                pin,
                   bool                pullUp,    //
                   bool                pullDown,  //
                   gpio_dir            output,    //
                   gpio_override       override,  //
                   gpio_function_t     function,  //
                   gpio_slew_rate      slewRate,  //
                   gpio_drive_strength strenght   //
    ) {
        if (NUM_BANK0_GPIOS <= pin) {
            throw std::runtime_error("Pin number out of range");
        }

        if (pullUp && pullDown) {
            throw std::runtime_error("Cannot set both pull-up and pull-down");
        }

        gpio_set_dir(pin, output);
        gpio_set_function(pin, function);
        gpio_set_slew_rate(pin, slewRate);
        gpio_set_pulls(pin, pullUp, pullDown);

        gpio_set_inover(pin, override);
        gpio_set_outover(pin, override);
        gpio_set_irqover(pin, override);
    }

    void setPinLevel(uint pin, bool enabled) {
        gpio_put(pin, enabled);
    }

    bool getPinLevel(uint pin) {
        return gpio_get(pin);
    }

    void writeAll(uint64_t value, uint64_t mask) {
        gpio_put_masked64(mask, value);
    }

    uint64_t readAll() {
        return gpio_get_all64();
    }

    gpio_dir getPindir(uint pin) {
        return static_cast<gpio_dir>(gpio_get_dir(pin));
    }

    gpio_drive_strength getDriveStrength(uint pin) {
        return gpio_get_drive_strength(pin);
    }

    gpio_slew_rate getSlewRate(uint pin) {
        return gpio_get_slew_rate(pin);
    }

}  // namespace corekit::Gpio