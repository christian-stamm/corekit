#pragma once

#include <FreeRTOS.h>
#include <semphr.h>

#include <cstdint>
#include <memory>

namespace corekit::platform {

    class Semaphore {
       public:
        using Ptr = std::shared_ptr<Semaphore>;

        struct Opset {
            explicit Opset(SemaphoreHandle_t& handle);
            virtual ~Opset();

            virtual bool take(TickType_t ticks = portMAX_DELAY) const;
            virtual bool release() const;

           protected:
            SemaphoreHandle_t& handle_;
        };

        using CoreSet = Opset;

        struct IsrSet : public Opset {
            explicit IsrSet(SemaphoreHandle_t& handle);
            ~IsrSet() override = default;

            virtual bool take(TickType_t) const override;
            virtual bool release() const override;
        };

        Semaphore(uint32_t initial_count = 0, uint32_t max_count = 1);
        ~Semaphore();

        Semaphore(const Semaphore&)            = delete;
        Semaphore(Semaphore&&)                 = delete;
        Semaphore& operator=(const Semaphore&) = delete;
        Semaphore& operator=(Semaphore&&)      = delete;

        void acquire();
        void release();
        bool try_acquire();

       private:
        inline void acquire(const Opset& opset) {
            opset.take();
        }

        inline void release(const Opset& opset) {
            opset.release();
        }

        inline bool try_acquire(const Opset& opset) {
            return opset.take(0);
        }

        SemaphoreHandle_t semaphore_;
        CoreSet           core_set_;
        IsrSet            isr_set_;
    };

}  // namespace corekit::platform