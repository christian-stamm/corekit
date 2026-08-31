#pragma once

#include <hardware/gpio.h>

#include <cstdint>
#include <functional>
#include <set>

#include "corekit/result.hpp"
#include "corekit/stoptoken.hpp"
#include "corekit/task.hpp"

namespace corekit::Gpio {

    constexpr uint64_t UNMASKED = 0xFFFFFFFFFFFFFFFF;

    using Pin                   = uint8_t;
    using Set                   = std::set<Pin>;
    using Handle                = std::function<void(uint pin, uint32_t events)>;

    struct Range {
            Range(Pin base = 0, uint length = 1);

            Pin operator[](int index) const;

            Pin lower() const;
            Pin upper() const;

            size_t count() const { return length; }

            uint64_t mask() const;
            Range    slice(int shift, int length) const;
            Set      pins() const;

        private:

            Pin  base;
            uint length;
    };

    extern bool setChannelIRQ(uint pin, uint32_t events = GPIO_IRQ_EDGE_RISE | GPIO_IRQ_EDGE_FALL, Handle callback = nullptr);

    extern VoidResult configure(
        uint pin,
        bool pullUp                  = false,                  //
        bool pullDown                = false,                  //
        gpio_dir output              = GPIO_OUT,               //
        gpio_override override       = GPIO_OVERRIDE_NORMAL,   //
        gpio_function_t function     = GPIO_FUNC_SIO,          //
        gpio_slew_rate slewRate      = GPIO_SLEW_RATE_SLOW,    //
        gpio_drive_strength strenght = GPIO_DRIVE_STRENGTH_2MA //
    );

    extern void setPinLevel(uint pin, bool enabled);
    extern bool getPinLevel(uint pin);

    extern void     writeAll(uint64_t value, uint64_t mask = UNMASKED);
    extern uint64_t readAll();

    extern gpio_dir            getPindir(uint pin);
    extern gpio_drive_strength getDriveStrength(uint pin);
    extern gpio_slew_rate      getSlewRate(uint pin);

    class IsrDaemon : public Task {
        public:

            using Ptr = std::shared_ptr<IsrDaemon>;

            static Ptr get() {
                static Ptr daemon = Ptr(new IsrDaemon());
                return daemon;
            }

            void spin_once() const;

        protected:

            virtual VoidResult on_init(StopToken token) override;
            virtual VoidResult on_exec(StopToken token) override;
            virtual VoidResult on_exit(StopToken token) override;

            void reconfigure();

        private:

            IsrDaemon();
    };

}; // namespace corekit::Gpio
