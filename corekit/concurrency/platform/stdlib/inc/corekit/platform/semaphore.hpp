#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>

namespace corekit::platform {

    class Semaphore {
       public:
        using Ptr = std::shared_ptr<Semaphore>;

        explicit Semaphore(std::size_t init_count = 0,
                           std::size_t max_count  = 1);

        void acquire();

        [[nodiscard]]
        bool try_acquire();

        void release();

        [[nodiscard]]
        std::size_t max_count() const noexcept;

       private:
        mutable std::mutex      mutex_;
        std::condition_variable cv_;

        std::size_t       count_;
        const std::size_t max_count_;
    };

}  // namespace corekit::platform