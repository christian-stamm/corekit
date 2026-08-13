#pragma once
#include <semaphore>

#include "corekit/iface/smphr.hpp"

namespace corekit {

    class PosixSemaphore : public ISemaphore {
       public:
        using Ptr = std::shared_ptr<PosixSemaphore>;

        virtual void acquire() override {
            m_semaphore.acquire();
        }

        virtual void release() override {
            m_semaphore.release();
        }

        virtual bool try_acquire() override {
            return m_semaphore.try_acquire();
        }

       private:
        std::counting_semaphore<> m_semaphore;
    };

    using Semaphore = PosixSemaphore;

}  // namespace corekit
