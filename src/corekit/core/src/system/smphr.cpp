#include "corekit/system/smphr.hpp"

namespace corekit {
    namespace system {

        Semaphore::Semaphore(uint initial) : impl(new SemaphoreImpl(initial)) {}

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

    };  // namespace system
};      // namespace corekit
