#pragma once
#include <memory>
#include <semaphore>

namespace corekit {

    class PosixSemaphore {
       public:
        using Ptr = std::shared_ptr<PosixSemaphore>;

        PosixSemaphore(uint count = 0);

        void acquire();
        void release();
        bool try_acquire();

       private:
        std::counting_semaphore<> m_semaphore;
    };

}  // namespace corekit
