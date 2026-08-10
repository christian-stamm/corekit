#pragma once
#include <semaphore>

namespace corekit {

 class PosixSemaphore {

        public:

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

using Semaphore = PosixSemaphore;

}  // namespace corekit
