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

        Atomic(T value = T()) : value(value) {}

        Atomic(const Atomic&)            = delete;
        Atomic(Atomic&&)                 = delete;
        Atomic& operator=(const Atomic&) = delete;
        Atomic& operator=(Atomic&&)      = delete;

        const T& load() const {
            taskENTER_CRITICAL();
            const T& result = value;
            taskEXIT_CRITICAL();
            return result;
        }

        void store(T value) {
            taskENTER_CRITICAL();
            this->value = value;
            taskEXIT_CRITICAL();
        }

        bool compare_exchange(T& expected, T desired) {
            taskENTER_CRITICAL();

            bool exchanged = false;
            if (value == expected) {
                value     = desired;
                exchanged = true;
            } else {
                expected = value;
            }

            taskEXIT_CRITICAL();
            return exchanged;
        }

       private:
        T value;
    };

    extern template class Atomic<bool>;

}  // namespace corekit::platform