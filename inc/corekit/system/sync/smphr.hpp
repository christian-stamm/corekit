#pragma once
#include <concepts>

#include "corekit/types.hpp"

#if defined(PLATFORM_PICO)
#    include "platform/pico/smphr.hpp"
#elif defined(PLATFORM_UNIX)
#    include "platform/unix/smphr.hpp"
#endif

namespace corekit {
    namespace system {

        using namespace corekit::types;

        template <typename T>
        concept SemaphoreConcept = requires(T t, uint count, float time) {
            { t.try_acquire_until(count, time) } -> std::same_as<bool>;
            { t.release(count) } -> std::same_as<void>;
        };

        static_assert(SemaphoreConcept<SemaphoreImpl>);

        class Semaphore {
           public:
            Semaphore(uint initial = 0);
            ~Semaphore();

            void acquire(uint count = 1);
            void release(uint count = 1);
            bool try_acquire(uint count = 1);
            bool try_acquire_for(uint count = 1, float secs = 0.0f);
            bool try_acquire_until(uint count, float time);

           private:
            SemaphoreImpl* impl;
        };

    };  // namespace system
};      // namespace corekit
