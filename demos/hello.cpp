#include <pico/stdlib.h>

#include <format>

#include "corekit/executor.hpp"
#include "corekit/gpiodevice.hpp"
#include "corekit/logger.hpp"
#include "corekit/task.hpp"
#include "corekit/time.hpp"
#include "corekit/uartdevice.hpp"

using namespace corekit;

class GpioTask : public Task {
   public:
    virtual VoidResult on_enter(StopToken token) override {
        return VoidResult();
    }

    virtual VoidResult on_run(StopToken token) override {
        for (uint i = 0; i < 8; ++i) {
            Gpio::configure(i,
                            false,
                            false,
                            GPIO_OUT,
                            GPIO_OVERRIDE_NORMAL,
                            GPIO_FUNC_SIO);
        }

        uint64_t value = 0;
        while (!token.stop_requested()) {
            if (value == 0) {
                value = 1;
            }

            Gpio::writeAll(value, 0xFF);
            Time::sleep(1);

            value <<= 1;
            value &= 0xFF;

            logger_.info() << std::format("GpioTask: value {:08b}", value);
        }

        return VoidResult();
    }

    virtual VoidResult on_leave(StopToken token) override {
        return VoidResult();
    }

   private:
    Logger logger_{"GpioTask"};
};

int main() {
    sleep_ms(10);  // Stabilize
    Logging::reconfigure(std::make_shared<Uart::Device>());

    Executor      executor(2, 2);
    GpioTask::Ptr gpioTask = std::make_shared<GpioTask>();

    executor.enqueue(gpioTask);
    executor.launch();

    return 0;
}