#include "corekit/platform/conditionvariable.hpp"

#include "corekit/check.hpp"

namespace corekit::platform {

    ConditionVariable::ConditionVariable(uint32_t max_waiters)
        : waiters_(max_waiters) { }

    ConditionVariable::~ConditionVariable() { corecheck(waiters_.empty(), RuntimeError("ConditionVariable destroyed with waiters still present")); }

    void ConditionVariable::wait(std::unique_lock<Mutex>& lock) {
        Semaphore waiter(0, 1);

        corecheck(!xPortIsInsideInterrupt());

        // Register ourselves while still holding the caller's mutex.
        //
        // Do not block here. Blocking while holding `lock` could deadlock
        // because another thread may need the same mutex in order to notify.
        //

        corecheck(waiters_.push(&waiter));
        //
        // The waiter is now visible to notify_one()/notify_all().
        //
        // If notification happens between unlock() and xSemaphoreTake(),
        // the binary semaphore remembers the wakeup.
        //
        lock.unlock();

        waiter.acquire();

        lock.lock();
    }

    void ConditionVariable::notify_one() {
        Semaphore* waiter = nullptr;

        if (waiters_.pop(waiter)) {
            corecheck(waiter != nullptr);
            waiter->release();
        }
    }

    void ConditionVariable::notify_all() {
        const UBaseType_t count = waiters_.size();

        for (UBaseType_t i = 0; i < count; ++i) { notify_one(); }
    }

} // namespace corekit::platform