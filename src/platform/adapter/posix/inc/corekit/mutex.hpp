#pragma once
#include <mutex>

#include "corekit/iface/mutex.hpp"

namespace corekit {

    class PosixMutex : public IMutex {
       public:
        using Ptr = std::shared_ptr<PosixMutex>;

        virtual void lock() override {
            m_mutex.lock();
        }

        virtual void unlock() override {
            m_mutex.unlock();
        }

        virtual bool try_lock() override {
            return m_mutex.try_lock();
        }

       private:
        std::mutex m_mutex;
    };

    using Mutex = PosixMutex;

}  // namespace corekit
