#pragma once
#include <hardware/gpio.h>

#include <cstdint>
#include <functional>

namespace corekit::Gpio {

    static constexpr uint64_t UNMASKED = 0xFFFFFFFFFFFFFFFF;
    using Handle = std::function<void(uint, gpio_irq_level)>;

    static void enableIRQ();
    static void disableIRQ();
    static void setHandle(uint pin, gpio_irq_level event, Handle&& callback);

    extern void configure(
        uint                pin,
        bool                pullUp   = false,                   //
        bool                pullDown = false,                   //
        gpio_dir            output   = GPIO_OUT,                //
        gpio_override       override = GPIO_OVERRIDE_NORMAL,    //
        gpio_function_t     function = GPIO_FUNC_SIO,           //
        gpio_slew_rate      slewRate = GPIO_SLEW_RATE_SLOW,     //
        gpio_drive_strength strenght = GPIO_DRIVE_STRENGTH_2MA  //
    );

    extern void setPinLevel(uint pin, bool enabled);
    extern bool getPinLevel(uint pin);

    extern void     writeAll(uint64_t value, uint64_t mask = UNMASKED);
    extern uint64_t readAll();

    extern gpio_dir            getPindir(uint pin);
    extern gpio_drive_strength getDriveStrength(uint pin);
    extern gpio_slew_rate      getSlewRate(uint pin);

};  // namespace corekit::Gpio
