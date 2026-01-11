#include <stdexcept>

namespace corekit {
    namespace system {
        namespace concurrency {

            struct SemaphoreImpl {
                SemaphoreImpl(uint initial) {}

                void acquire(uint count) {
                    throw std::runtime_error(
                        "Semaphore acquire not implemented on pico platform");
                }

                void release(uint count) {
                    throw std::runtime_error(
                        "Semaphore release not implemented on pico platform");
                }

                bool try_acquire(uint count) {
                    return false;
                }

                bool try_acquire_for(uint count, float secs) {
                    return false;
                }

                bool try_acquire_until(uint count, float time) {
                    return false;
                }
            };

        };  // namespace concurrency
    };  // namespace system
};  // namespace corekit
