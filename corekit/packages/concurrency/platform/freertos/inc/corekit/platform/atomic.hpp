#pragma once
#include <FreeRTOS.h>
#include <task.h>

#include <memory>

namespace corekit::platform {

    template <typename T>
    class Atomic {
       public:
        using Ptr       = std::shared_ptr<Atomic<T>>;
        using ValueType = T;

        struct Opset {
            virtual inline UBaseType_t enter() const {
                vTaskEnterCritical();
                return 0;
            }

            virtual inline void exit(UBaseType_t) const {
                vTaskExitCritical();
            }
        };

        using CoreOpset = Opset;

        struct IsrOpset : public Opset {
            inline UBaseType_t enter() const override {
                return vTaskEnterCriticalFromISR();
            }

            inline void exit(UBaseType_t state) const override {
                vTaskExitCriticalFromISR(state);
            }
        };

        Atomic(T value = T()) : value(value) {}

        Atomic(const Atomic &)            = delete;
        Atomic(Atomic &&)                 = delete;
        Atomic &operator=(const Atomic &) = delete;
        Atomic &operator=(Atomic &&)      = delete;

        const T &load() const {
            return load(xPortIsInsideInterrupt() ? isr_set_ : core_set_);
        }

        void store(T value) {
            store(xPortIsInsideInterrupt() ? isr_set_ : core_set_, value);
        }

        bool compare_exchange(T &expected, T desired) {
            return compare_exchange(
                xPortIsInsideInterrupt() ? isr_set_ : core_set_,
                expected,
                desired);
        }

       private:
        inline const T &load(const Opset &opset) const {
            const UBaseType_t state  = opset.enter();
            const T          &result = value;
            opset.exit(state);
            return result;
        }

        inline void store(const Opset &opset, T value) {
            const UBaseType_t state = opset.enter();
            this->value             = value;
            opset.exit(state);
        }

        inline bool compare_exchange(const Opset &opset,
                                     T           &expected,
                                     T            desired) {
            const UBaseType_t state = opset.enter();

            bool exchanged = false;
            if (value == expected) {
                value     = desired;
                exchanged = true;
            } else {
                expected = value;
            }

            opset.exit(state);
            return exchanged;
        }

        T value;

        CoreOpset core_set_;
        IsrOpset  isr_set_;
    };

    extern template class Atomic<bool>;

}  // namespace corekit::platform