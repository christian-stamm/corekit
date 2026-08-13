#pragma once
#include <memory>
#include <semaphore>

namespace corekit {

    class PosixSemaphore {
       public:
        using Ptr = std::shared_ptr<PosixSemaphore>;

        PosixSemaphore(uint count = 0) : m_semaphore(count) {}

        void acquire() {
            m_semaphore.acquire();
        }

        void release() {
            m_semaphore.release();
        }

        bool try_acquire() {
            return m_semaphore.try_acquire();
        }

       private:
        std::counting_semaphore<> m_semaphore;
    };

}  // namespace corekit
