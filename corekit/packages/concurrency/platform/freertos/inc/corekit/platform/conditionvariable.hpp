#pragma once

#include <FreeRTOS.h>
#include <task.h>

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "corekit/mutex.hpp"

namespace corekit::platform {

    class ConditionVariable {
       public:
        ConditionVariable() = default;

        ConditionVariable(const ConditionVariable&)            = delete;
        ConditionVariable(ConditionVariable&&)                 = delete;
        ConditionVariable& operator=(const ConditionVariable&) = delete;
        ConditionVariable& operator=(ConditionVariable&&)      = delete;

        template <typename Predicate>
        void wait(std::unique_lock<Mutex>& lock, Predicate predicate) {
            while (!predicate()) {
                wait(lock);
            }
        }

        void wait(std::unique_lock<Mutex>& lock);
        void notify_one();
        void notify_all();

       private:
        Mutex                    waiters_mutex_;
        std::deque<TaskHandle_t> waiters_;
    };

}  // namespace corekit::platform