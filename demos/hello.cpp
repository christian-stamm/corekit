#include <hardware/pio.h>
#include <pico/stdlib.h>

#include <memory>

#include "corekit/executor.hpp"
#include "corekit/logger.hpp"
#include "corekit/piodevice.hpp"
#include "corekit/platform/piodevice.hpp"
#include "corekit/task.hpp"
#include "corekit/time.hpp"
#include "corekit/uartdevice.hpp"
#include "loopback.hpp"

using namespace corekit;

class LoopbackProgram : public Pio::Program {
   public:
    LoopbackProgram() : Pio::Program(loopback_program) {}

    virtual Pio::NodeConf buildNodeConf(PIO  block,
                                        uint node,
                                        uint base) override {
        Pio::NodeConf ncfg = loopback_program_get_default_config(base);
        sm_config_set_out_shift(&ncfg, true, true, 32);
        sm_config_set_in_shift(&ncfg, false, true, 8);
        return ncfg;
    }
};

class PioTask : public Task {
   public:
    PioTask(PIO block)
        : pio_device_(Pio::Node<uint32_t>::requestUnused(block))
        , program_(std::make_shared<LoopbackProgram>()) {}

    virtual VoidResult on_enter(StopToken token) override {
        pio_device_->deploy(program_);
        return VoidResult();
    }

    virtual VoidResult on_run(StopToken token) override {
        while (!token.stop_requested()) {
            const uint32_t send = 0x12345678;
            uint32_t       recv = 0;
            logger_.info() << std::format("Sending: 0x{:08X}", send);
            pio_device_->write(send);
            pio_device_->read(recv);
            logger_.info() << std::format("Received: 0x{:08X}", recv);
            Time::sleep(1);
        }

        return VoidResult();
    }

    virtual VoidResult on_leave(StopToken token) override {
        pio_device_->unload();
        return VoidResult();
    }

   private:
    Logger                   logger_{"PioTask"};
    Pio::Program::Ptr        program_;
    Pio::Node<uint32_t>::Ptr pio_device_;
};

int main() {
    sleep_ms(10);  // Stabilize
    Logging::reconfigure(std::make_shared<UartDevice>());

    Executor     executor(2, 2);
    PioTask::Ptr pioTask = std::make_shared<PioTask>(pio0);

    executor.enqueue(pioTask);
    executor.launch();

    return 0;
}