#include "corekit/platform/semaphore.hpp"

#include <cassert>

namespace corekit::platform {

    Semaphore::Semaphore(const std::size_t init_count,
                         const std::size_t max_count)
        : count_(init_count)
        , max_count_(max_count) {
        assert(max_count_ > 0);
        assert(count_ <= max_count_);
    }

    void Semaphore::acquire() {
        std::unique_lock lock{mutex_};

        cv_.wait(lock, [this] { return count_ > 0; });

        --count_;
    }

    bool Semaphore::try_acquire() {
        std::lock_guard lock{mutex_};

        if (count_ == 0)
            return false;

        --count_;
        return true;
    }

    void Semaphore::release() {
        {
            std::lock_guard lock{mutex_};

            assert(count_ < max_count_);

            if (count_ >= max_count_)
                return;

            ++count_;
        }

        cv_.notify_one();
    }

    std::size_t Semaphore::max_count() const noexcept {
        return max_count_;
    }

}  // namespace corekit::platform