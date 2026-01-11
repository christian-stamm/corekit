#pragma once
#include <concepts>

namespace corekit {
    namespace system {
        namespace concurrency {

            struct MutexImpl;

            template <typename T>
            concept MutexConcept = requires(T t) {
                { t.lock() } -> std::same_as<void>;
                { t.unlock() } -> std::same_as<void>;
                { t.try_lock() } -> std::same_as<bool>;
            };

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

        };  // namespace concurrency
    };  // namespace system
};  // namespace corekit
