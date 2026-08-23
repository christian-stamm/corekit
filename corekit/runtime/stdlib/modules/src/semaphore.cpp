

#include <stdexcept>

#include "corekit/semaphore.hpp"

namespace corekit {

    Semaphore::Semaphore(uint64_t initial, uint64_t limit)
        : count_(initial)
        , limit_(limit) {
        if (limit <= 0) {
            throw std::invalid_argument("Limit must be greater than zero");
        }

        if (limit < initial) {
            throw std::invalid_argument("Initial count cannot exceed limit");
        }
    }

    bool Semaphore::try_acquire() {
        std::unique_lock lock(mutex_);
        return try_acquire_unsafe();
    }

    void Semaphore::acquire() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return try_acquire_unsafe(); });
    }

    void Semaphore::release() {
        std::unique_lock lock(mutex_);

        if (count_ == limit_) {
            // Overflow policy: assert/throw/etc.
            return;
        }

        ++count_;
        cv_.notify_one();
    }

    bool Semaphore::try_acquire_unsafe() {
        if (count_ == 0) {
            return false;
        }

        --count_;
        return true;
    }

}  // namespace corekit