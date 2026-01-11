#include "corekit/system/concurrency/mutex.hpp"

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

            static_assert(MutexConcept<MutexImpl>);

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
