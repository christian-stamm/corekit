#include <hardware/pio.h>
#include <pico/stdlib.h>

#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <vector>

#include "corekit/dmadevice.hpp"
#include "corekit/executor.hpp"
#include "corekit/logger.hpp"
#include "corekit/piodevice.hpp"
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

class DmaTask : public Task {
   public:
    DmaTask() : dma_device_(Dma::Device::requestUnused()) {
        Dma::Device::enableIRQ();

        aligned_src_ptr_ = new uint32_t(0);
        aligned_dst_ptr_ = static_cast<uint32_t*>(
            aligned_alloc(256 * sizeof(uint32_t), 256 * sizeof(uint32_t)));
    }

    virtual VoidResult on_enter(StopToken token) override {
        dma_transfer_ = std::make_shared<Dma::Transfer>(
            0,
            DREQ_FORCE,
            reinterpret_cast<volatile void*>(aligned_src_ptr_),
            reinterpret_cast<volatile void*>(aligned_dst_ptr_),
            DMA_ADDRESS_UPDATE_NONE,
            DMA_ADDRESS_UPDATE_INCREMENT,
            256,
            Dma::XferSize::DMA_SIZE_32,
            Dma::Wrapping::Write,
            false,
            false,
            -1,
            nullptr);

        return VoidResult();
    }

    virtual VoidResult on_run(StopToken token) override {
        if (!dma_device_->process(dma_transfer_)) {
            logger_.warn() << "DMA device is busy.";
        }

        uint32_t counter = 0;
        while (!token.stop_requested()) {
            *aligned_src_ptr_ = counter++;

            Time::sleep(1);
            logger_() << aligned_src_ptr_[0] << " -> " << aligned_dst_ptr_[0];
        }

        return VoidResult();
    }

    virtual VoidResult on_leave(StopToken token) override {
        return VoidResult();
    }

   private:
    Logger             logger_{"DmaTask"};
    Dma::Device::Ptr   dma_device_;
    Dma::Transfer::Ptr dma_transfer_;

    uint32_t* aligned_src_ptr_ = nullptr;
    uint32_t* aligned_dst_ptr_ = nullptr;
};

int main() {
    sleep_ms(10);  // Stabilize
    Logging::reconfigure(std::make_shared<Uart::Device>());

    Executor     executor(2, 2);
    DmaTask::Ptr dmaTask = std::make_shared<DmaTask>();
    PioTask::Ptr pioTask = std::make_shared<PioTask>(pio0);

    executor.enqueue(pioTask);
    executor.enqueue(dmaTask);
    executor.launch();

    return 0;
}