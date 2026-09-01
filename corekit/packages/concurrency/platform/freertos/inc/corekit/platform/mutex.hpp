#pragma once

#include <FreeRTOS.h>
#include <semphr.h>

#include <memory>

namespace corekit::platform {

    class Mutex {
       public:
        using Ptr = std::shared_ptr<Mutex>;

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

        Mutex();
        ~Mutex();

        Mutex(const Mutex&)            = delete;
        Mutex(Mutex&&)                 = delete;
        Mutex& operator=(const Mutex&) = delete;
        Mutex& operator=(Mutex&&)      = delete;

        void lock();
        void unlock();
        bool try_lock();

       private:
        inline void lock(const Opset& opset) {
            opset.take();
        }

        inline void unlock(const Opset& opset) {
            opset.release();
        }

        inline bool try_lock(const Opset& opset) {
            return opset.take(0);
        }

        SemaphoreHandle_t handle_;
        CoreSet           core_set_;
        IsrSet            isr_set_;
    };

}  // namespace corekit::platform