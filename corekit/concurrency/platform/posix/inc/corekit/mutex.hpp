#pragma once
#include <memory>
#include <mutex>

namespace corekit {

    class PosixMutex {
       public:
        using Ptr = std::shared_ptr<PosixMutex>;

        void lock();
        void unlock();
        bool try_lock();

       private:
        std::mutex m_mutex;
    };

}  // namespace corekit
