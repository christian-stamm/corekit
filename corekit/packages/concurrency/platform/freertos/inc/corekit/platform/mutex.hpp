#pragma once

#include <FreeRTOS.h>
#include <semphr.h>

#include <memory>
#include <optional>

namespace corekit::platform {

    class Mutex {
        public:

            using Ptr     = std::shared_ptr<Mutex>;
            using Timeout = std::optional<double>; // seconds

            Mutex();
            ~Mutex();

            Mutex(const Mutex&)            = delete;
            Mutex(Mutex&&)                 = delete;
            Mutex& operator=(const Mutex&) = delete;
            Mutex& operator=(Mutex&&)      = delete;

            void lock();
            void unlock();
            bool try_lock(Timeout seconds = std::nullopt);

        private:

            StaticSemaphore_t storage_;
            SemaphoreHandle_t handle_;
    };

} // namespace corekit::platform