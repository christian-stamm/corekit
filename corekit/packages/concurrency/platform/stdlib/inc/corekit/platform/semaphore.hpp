#pragma once

#include <limits>
#include <memory>
#include <mutex>

#include "corekit/platform/conditionvariable.hpp"

namespace corekit::platform {

    class Semaphore {
       public:
        using Ptr = std::shared_ptr<Semaphore>;

        Semaphore(
            std::size_t init_count = 0,
            std::size_t max_count  = std::numeric_limits<std::size_t>::max());

        void release();
        void acquire();

        [[nodiscard]]
        bool try_acquire();

        const std::size_t max_count_;

       private:
        mutable std::mutex mutex_;
        ConditionVariable  cv_;

        std::size_t count_;
    };

}  // namespace corekit::platform