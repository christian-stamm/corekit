#pragma once
#include <memory>
#include <FreeRTOS.h>
#include <task.h>

namespace corekit {

    template <typename T>
    class FreeRTOSAtomic {
       public:
        using Ptr       = std::shared_ptr<FreeRTOSAtomic<T>>;
        using ValueType = T;

        FreeRTOSAtomic(T value = T()) : value(value) {}

        const T& load() const {
            auto state = enter_critical();
            const T& return_value = value;
            exit_critical(state);
            return return_value;
        }

        void store(T value) {
            auto state = enter_critical();
            this->value = value;
            exit_critical(state);
        }

        bool compare_exchange_strong(T& expected, T desired) {
            auto state = enter_critical();
            if (value == expected) {
                value = desired;
                return true;
            } else {
                expected = value;
                return false;
            }
            exit_critical(state);
        }

       private:
        static bool in_isr()
        {
            return xPortIsInsideInterrupt() != pdFALSE;
        }

        static UBaseType_t enter_critical()
        {
            if (in_isr()) {
                return taskENTER_CRITICAL_FROM_ISR();
            }

            taskENTER_CRITICAL();
            return 0;
        }

        static void exit_critical(UBaseType_t state)
        {
            if (in_isr()) {
                taskEXIT_CRITICAL_FROM_ISR(state);
            } else {
                taskEXIT_CRITICAL();
            }
        }

        T value;
    };

}  // namespace corekit