#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>

namespace corekit {

    class Semaphore {
       public:
        using Ptr = std::shared_ptr<Semaphore>;

        Semaphore(uint64_t initial = 0, uint64_t limit = 1);

        void acquire();
        void release();
        bool try_acquire();

       private:
        bool try_acquire_unsafe();

        uint64_t count_;
        uint64_t limit_;

        std::mutex              mutex_;
        std::condition_variable cv_;
    };

}  // namespace corekit
