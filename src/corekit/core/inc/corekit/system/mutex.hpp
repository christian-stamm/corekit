#pragma once
#include <concepts>

#if defined(PLATFORM_PICO)
#    include "corekit/platform/pico/mutex.hpp"
#elif defined(PLATFORM_UNIX)
#    include "corekit/platform/unix/mutex.hpp"
#endif

namespace corekit {
    namespace system {

        template <typename T>
        concept MutexConcept = requires(T t) {
            { t.lock() } -> std::same_as<void>;
            { t.unlock() } -> std::same_as<void>;
            { t.try_lock() } -> std::same_as<bool>;
        };

        static_assert(MutexConcept<MutexImpl>);

        class Mutex {
           public:
            Mutex();
            ~Mutex();

            Mutex(const Mutex&)             = delete;
            Mutex(const Mutex&&)            = delete;
            Mutex& operator=(const Mutex&)  = delete;
            Mutex& operator=(const Mutex&&) = delete;

            void lock();
            void unlock();
            bool try_lock();

           private:
            MutexImpl* impl;
        };

    };  // namespace system
};      // namespace corekit
