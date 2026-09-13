#include "corekit/semaphore.hpp"

#include "corekit/check.hpp"

namespace corekit {

    Semaphore::Semaphore(const std::size_t initial_count, const std::size_t max_count)
        : count_(initial_count)
        , max_count_(max_count) {
        core::check(0 < max_count_, RuntimeError("Semaphore::Semaphore() max_count must be greater than zero"));
        core::check(
            count_ <= max_count_,
            RuntimeError("Semaphore::Semaphore() init_count must be less than or equal to max_count"));
    }

    void Semaphore::acquire() {
        std::unique_lock lock{mutex_};

        cv_.wait(lock, [this] { return count_ > 0; });

        --count_;
    }

    bool Semaphore::try_acquire() {
        std::lock_guard lock{mutex_};

        if (count_ == 0) { return false; }

        --count_;
        return true;
    }

    void Semaphore::release() {
        {
            std::lock_guard lock{mutex_};

            assert(count_ < max_count_);

            if (count_ >= max_count_) { return; }

            ++count_;
        }

        cv_.notify_one();
    }

} // namespace corekit