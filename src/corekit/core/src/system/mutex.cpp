#include "corekit/system/mutex.hpp"

namespace corekit {
    namespace system {

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

    };  // namespace system
};      // namespace corekit
