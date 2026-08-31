#include "corekit/gpiodevice.hpp"

#include <hardware/gpio.h>
#include <hardware/irq.h>
#include <pico/types.h>

#include <format>

#include "corekit/error.hpp"
#include "corekit/math.hpp"
#include "corekit/semaphore.hpp"

namespace corekit::Gpio {

    constexpr uint8_t ALL_EVENTS = GPIO_IRQ_EDGE_RISE | GPIO_IRQ_EDGE_FALL | GPIO_IRQ_LEVEL_HIGH | GPIO_IRQ_LEVEL_LOW;

    struct Event {
            Atomic<uint8_t> irq_events{0};
            uint8_t         usr_events = 0;
            Handle          callback   = nullptr;
    };

    std::array<Event, NUM_BANK0_GPIOS> irq_events{};
    Semaphore                          irq_trigger{0, 1};
    Atomic<bool>                       cb_modified{false};

    extern "C" {
    __isr void shared_gpio_irq_callback(uint gpio, uint32_t flags) {
        if (gpio < NUM_BANK0_GPIOS) {
            irq_events[gpio].irq_events.fetch_or(static_cast<uint8_t>(flags));
            irq_trigger.release();
        }
    }
    }

    void enableIRQ() {
        irq_set_enabled(IO_IRQ_BANK0, false);

        for (uint8_t pin = 0; pin < NUM_BANK0_GPIOS; ++pin) {
            gpio_set_irq_enabled(pin, ALL_EVENTS, false);
            gpio_acknowledge_irq(pin, ALL_EVENTS);
            irq_events[pin].irq_events.store(0);
        }

        gpio_set_irq_callback(shared_gpio_irq_callback);
        irq_set_enabled(IO_IRQ_BANK0, true);
    }

    void disableIRQ() {
        irq_set_enabled(IO_IRQ_BANK0, false);
        gpio_set_irq_callback(nullptr);

        for (uint8_t pin = 0; pin < NUM_BANK0_GPIOS; ++pin) {
            gpio_acknowledge_irq(pin, ALL_EVENTS);
            irq_events[pin].irq_events.store(0);
            irq_events[pin].callback   = nullptr;
            irq_events[pin].usr_events = 0;
        }
    }

    // -----------------------------------------------------------------
    // Daemon
    // -----------------------------------------------------------------

    IsrDaemon::IsrDaemon()
        : Task("GpioDaemon") { }

    VoidResult IsrDaemon::on_init(StopToken token) {
        enableIRQ();

        cb_modified.store(true);
        this->reconfigure();

        return VoidResult();
    }

    VoidResult IsrDaemon::on_exec(StopToken token) {
        while (!token.stop_requested()) {
            irq_trigger.acquire();
            this->reconfigure();

            for (uint pin = 0; pin < NUM_BANK0_GPIOS; ++pin) {
                Event& event         = irq_events[pin];

                const uint8_t bits   = event.irq_events.exchange(0);
                const bool    notify = (bits & event.usr_events) != 0;
                const Handle& handle = event.callback;

                if (notify && handle) { handle(pin, bits); }
            }
        }

        return VoidResult();
    }

    VoidResult IsrDaemon::on_exit(StopToken token) {
        disableIRQ();
        return VoidResult();
    }

    void IsrDaemon::reconfigure() {
        if (!cb_modified.exchange(false)) { return; }

        for (uint pin = 0; pin < NUM_BANK0_GPIOS; ++pin) {
            const Event& event   = irq_events[pin];
            const bool   enabled = (event.callback != nullptr && event.usr_events != 0);

            gpio_set_irq_enabled(pin, ALL_EVENTS, false);

            if (enabled) {
                Gpio::configure(pin, false, false, GPIO_IN);
                gpio_acknowledge_irq(pin, ALL_EVENTS);
            }

            gpio_set_irq_enabled(pin, event.usr_events, enabled);
        }
    }

    void IsrDaemon::spin_once() const { irq_trigger.release(); }

    // -----------------------------------------------------------------
    // Range
    // -----------------------------------------------------------------

    Range::Range(Pin base, uint length)
        : base(base)
        , length(length) { }

    Pin Range::operator[](int index) const {
        using namespace corekit::math;
        return base + wrap(index, length);
    }

    Pin Range::lower() const { return base; }

    Pin Range::upper() const { return base + length - 1; }

    uint64_t Range::mask() const { return ((1ull << length) - 1) << base; }

    Range Range::slice(int shift, int length) const {
        Pin base                = (*this)[shift];

        const bool out_of_range = (length < 0) || (static_cast<int>(this->base + this->length) < (base + length));

        corecheck(!out_of_range, OutOfRangeError(std::format("Slice out of range: base={} " "length={} shift={} slice_length={}", this->base, this->length, shift, length)));

        return Range(base, length);
    }

    Set Range::pins() const {
        Set set;

        for (uint i = 0; i < length; ++i) { set.insert((*this)[i]); }

        return set;
    }

    bool setChannelIRQ(uint pin, uint32_t events, Handle callback) {
        corecheck(pin < NUM_BANK0_GPIOS, OutOfRangeError(std::format("Invalid GPIO pin: {} (max={})", pin, NUM_BANK0_GPIOS - 1)));

        Event& event     = irq_events[pin];
        event.usr_events = static_cast<uint8_t>(events);
        event.callback   = std::move(callback);
        cb_modified.store(true);
        irq_trigger.release();
        return true;
    }

    VoidResult configure(
        uint                pin,
        bool                pullUp,   //
        bool                pullDown, //
        gpio_dir            output,   //
        gpio_override       override, //
        gpio_function_t     function, //
        gpio_slew_rate      slewRate, //
        gpio_drive_strength strenght  //
    ) {
        if (NUM_BANK0_GPIOS <= pin) { return OutOfRangeError(std::format("Invalid GPIO pin: {} (max={})", pin, NUM_BANK0_GPIOS - 1)); }

        if (pullUp && pullDown) { return InvalidArgumentError(std::format("Invalid GPIO pin configuration: " "pullUp and pullDown cannot both be true")); }

        gpio_set_dir(pin, output);
        gpio_set_function(pin, function);
        gpio_set_slew_rate(pin, slewRate);
        gpio_set_pulls(pin, pullUp, pullDown);

        gpio_set_inover(pin, override);
        gpio_set_outover(pin, override);
        gpio_set_irqover(pin, override);
        return VoidResult();
    }

    void setPinLevel(uint pin, bool enabled) { gpio_put(pin, enabled); }

    bool getPinLevel(uint pin) { return gpio_get(pin); }

    void writeAll(uint64_t value, uint64_t mask) { gpio_put_masked64(mask, value); }

    uint64_t readAll() { return gpio_get_all64(); }

    gpio_dir getPindir(uint pin) { return static_cast<gpio_dir>(gpio_get_dir(pin)); }

    gpio_drive_strength getDriveStrength(uint pin) { return gpio_get_drive_strength(pin); }

    gpio_slew_rate getSlewRate(uint pin) { return gpio_get_slew_rate(pin); }

} // namespace corekit::Gpio