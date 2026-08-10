#pragma once
#include <mutex>


namespace corekit {

class PosixMutex {

        public:

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

using Mutex = PosixMutex;

}  // namespace corekit
