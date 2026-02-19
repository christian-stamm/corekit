#include <mutex>

namespace corekit {
    namespace system {

        struct MutexImpl {
            std::mutex mutex;

            void lock() {
                mutex.lock();
            }

            void unlock() {
                mutex.unlock();
            }

            bool try_lock() {
                return mutex.try_lock();
            }
        };

    };  // namespace system
};      // namespace corekit
