#include <semaphore>

namespace corekit {
    namespace system {

        struct SemaphoreImpl {
            std::counting_semaphore<> semaphore;

            SemaphoreImpl(uint initial) : semaphore(initial) {}

            void acquire(uint count) {
                for (uint i = 0; i < count; ++i) {
                    semaphore.acquire();
                }
            }

            void release(uint count) {
                semaphore.release(count);
            }

            bool try_acquire(uint count) {
                for (uint i = 0; i < count; ++i) {
                    if (!semaphore.try_acquire()) {
                        return false;
                    }
                }
                return true;
            }

            bool try_acquire_for(uint count, float secs) {
                return false;
            }

            bool try_acquire_until(uint count, float time) {
                return false;
            }
        };

    };  // namespace system
};      // namespace corekit
