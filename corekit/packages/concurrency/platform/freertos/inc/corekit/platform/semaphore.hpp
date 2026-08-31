#pragma once

#include <FreeRTOS.h>
#include <semphr.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace corekit::platform {

    class Semaphore {
        public:

            using Ptr     = std::shared_ptr<Semaphore>;
            using Timeout = std::optional<double>; // seconds

            Semaphore(uint32_t initial_count = 0, uint32_t max_count = 1);
            ~Semaphore();

            Semaphore(const Semaphore&)            = delete;
            Semaphore(Semaphore&&)                 = delete;
            Semaphore& operator=(const Semaphore&) = delete;
            Semaphore& operator=(Semaphore&&)      = delete;

            void acquire();
            void release();
            bool try_acquire(Timeout seconds = std::nullopt);

        private:

            StaticSemaphore_t storage_;
            SemaphoreHandle_t handle_;
    };

} // namespace corekit::platform