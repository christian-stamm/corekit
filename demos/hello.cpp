#include "corekit/assert.hpp"
#include "corekit/executor.hpp"
#include "corekit/logger.hpp"
#include "corekit/stoptoken.hpp"
#include "corekit/task.hpp"
#include "corekit/uartdevice.hpp"

using namespace corekit;

class HelloTask : public Task {
   public:
    virtual VoidResult on_run(StopToken token) override {
        logger_.info() << "Hello from HelloTask!";
        return VoidResult();
    }

   private:
    Logger logger_{"HelloTask"};
};

class WorldTask : public Task {
   public:
    virtual VoidResult on_run(StopToken token) override {
        logger_.info() << "Hello from WorldTask!";
        return VoidResult();
    }

   private:
    Logger logger_{"WorldTask"};
};

int main() {
    Logger   logger{"Main"};
    Executor executor(2, 5);

    UartDevice::Ptr uart_device = std::make_shared<UartDevice>();
    HelloTask::Ptr  hello_task  = std::make_shared<HelloTask>();
    WorldTask::Ptr  world_task  = std::make_shared<WorldTask>();

    Logging::reconfigure(uart_device);
    uart_device->load();

    logger.info() << "Starting executor...";
    executor.enqueue(hello_task);
    executor.enqueue(world_task);
    executor.launch();

    return 0;
}