#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include "corekit/mutex.hpp"
#include "corekit/queue.hpp"
#include "corekit/semaphore.hpp"

namespace corekit::platform {

    class ConditionVariable {
        public:

            using Ptr = std::shared_ptr<ConditionVariable>;

            explicit ConditionVariable(uint32_t max_waiters);
            ~ConditionVariable();

            ConditionVariable(const ConditionVariable&)            = delete;
            ConditionVariable(ConditionVariable&&)                 = delete;
            ConditionVariable& operator=(const ConditionVariable&) = delete;
            ConditionVariable& operator=(ConditionVariable&&)      = delete;

            template <typename Predicate>
            void wait(std::unique_lock<Mutex>& lock, Predicate predicate) {
                while (!predicate()) { wait(lock); }
            }

            void wait(std::unique_lock<Mutex>& lock);

            void notify_one();
            void notify_all();

        private:

            Queue<Semaphore*> waiters_;
    };

} // namespace corekit::platform