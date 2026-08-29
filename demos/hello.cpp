#include <pico/stdlib.h>

#include <memory>

#include "corekit/assert.hpp"
#include "corekit/executor.hpp"
#include "corekit/logger.hpp"
#include "corekit/task.hpp"
#include "corekit/time.hpp"
#include "corekit/uartdevice.hpp"

using namespace corekit;

class HelloTask : public Task {
   public:
    virtual VoidResult on_run(StopToken token) override {
        while (!token.stop_requested()) {
            logger_() << "HELLO";
            Time::sleep(1);
        }

        return VoidResult();
    }

   private:
    Logger logger_{"HelloTask"};
};

class WorldTask : public Task {
   public:
    virtual VoidResult on_run(StopToken token) override {
        while (!token.stop_requested()) {
            logger_() << "WORLD";
            Time::sleep(1);
        }

        return VoidResult();
    }

   private:
    Logger logger_{"WorldTask"};
};

int main() {
    sleep_ms(100);  // Wait for the UART to be ready

    StreamDevice::Ptr device = std::make_shared<UartDevice>();
    device->load();
    Logging::reconfigure(device);

    Executor       executor(2, 2);
    HelloTask::Ptr hello_task = std::make_shared<HelloTask>();
    WorldTask::Ptr world_task = std::make_shared<WorldTask>();

    executor.enqueue(hello_task);
    executor.enqueue(world_task);
    executor.launch();

    return 0;
}