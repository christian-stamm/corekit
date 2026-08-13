#pragma once
#include <memory>
#include <mutex>

namespace corekit {

    class PosixMutex {
       public:
        using Ptr = std::shared_ptr<PosixMutex>;

        void lock() {
            m_mutex.lock();
        }

        void unlock() {
            m_mutex.unlock();
        }

        bool try_lock() {
            return m_mutex.try_lock();
        }

       private:
        std::mutex m_mutex;
    };

}  // namespace corekit
