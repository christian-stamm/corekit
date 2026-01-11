#include <mutex>

namespace corekit {
    namespace system {
        namespace concurrency {

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

        };  // namespace concurrency
    };  // namespace system
};  // namespace corekit