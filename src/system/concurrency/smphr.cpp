#include "corekit/system/concurrency/smphr.hpp"

#include <semaphore>

namespace corekit {
    namespace system {
        namespace concurrency {

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

            static_assert(SemaphoreConcept<SemaphoreImpl>);

            Semaphore::Semaphore(uint initial)
                : impl(new SemaphoreImpl(initial)) {}

            Semaphore::~Semaphore() {
                delete impl;
            }

            void Semaphore::acquire(uint count) {
                impl->acquire(count);
            }

            void Semaphore::release(uint count) {
                impl->release(count);
            }

            bool Semaphore::try_acquire(uint count) {
                return impl->try_acquire(count);
            }

            bool Semaphore::try_acquire_for(uint count, float secs) {
                return impl->try_acquire_for(count, secs);
            }

            bool Semaphore::try_acquire_until(uint count, float time) {
                return impl->try_acquire_until(count, time);
            }

        };  // namespace concurrency
    };  // namespace system
};  // namespace corekit
