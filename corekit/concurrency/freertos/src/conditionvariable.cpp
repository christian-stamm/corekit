#include "corekit/conditionvariable.hpp"

#include "corekit/check.hpp"

namespace corekit {

    ConditionVariable::ConditionVariable(uint32_t max_waiters)
        : waiters_(max_waiters) { }

    ConditionVariable::~ConditionVariable() {
        core::check(waiters_.empty(), RuntimeError("ConditionVariable destroyed with waiters still present"));
    }

    void ConditionVariable::wait(std::unique_lock<Mutex>& lock) {
        Semaphore waiter(0, 1);

        core::check(!xPortIsInsideInterrupt());
        core::check(waiters_.push(&waiter));

        lock.unlock();
        waiter.acquire();
        lock.lock();
    }

    void ConditionVariable::notify_one() {
        Semaphore* waiter = nullptr;

        if (waiters_.pop(waiter)) {
            core::check(waiter != nullptr);
            waiter->release();
        }
    }

    void ConditionVariable::notify_all() {
        const UBaseType_t count = waiters_.size();
        for (UBaseType_t i = 0; i < count; ++i) { notify_one(); }
    }

} // namespace corekit