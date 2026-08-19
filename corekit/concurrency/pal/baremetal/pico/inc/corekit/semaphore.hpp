#pragma once
#include <memory>

#include "pico/sync.h"

namespace corekit {

    class PicoSemaphore {
       public:
        using Ptr = std::shared_ptr<PicoSemaphore>;

        PicoSemaphore(uint count = 0);

        void acquire();
        void release();
        bool try_acquire();

       private:
        semaphore_t m_semaphore;
    };

    using Semaphore = PicoSemaphore;

}  // namespace corekit
