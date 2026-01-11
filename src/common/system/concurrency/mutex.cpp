#include "corekit/system/concurrency/mutex.hpp"

namespace corekit {
    namespace system {
        namespace concurrency {

            Mutex::Mutex() : impl(new MutexImpl()) {}

            Mutex::~Mutex() {
                delete impl;
            }

            void Mutex::lock() {
                impl->lock();
            }

            void Mutex::unlock() {
                impl->unlock();
            }

            bool Mutex::try_lock() {
                return impl->try_lock();
            }

        };  // namespace concurrency
    };  // namespace system
};  // namespace corekit
