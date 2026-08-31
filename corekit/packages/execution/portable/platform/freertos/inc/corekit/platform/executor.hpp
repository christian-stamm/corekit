#pragma once

#include <FreeRTOS.h>
#include <task.h>

#include <memory>
#include <vector>

#include "corekit/atomic.hpp"
#include "corekit/semaphore.hpp"
#include "corekit/stoptoken.hpp"
#include "corekit/task.hpp"

namespace corekit::platform {

    class Thread {
        public:

            struct LaunchArgs {
                    uint coremask = 0b1111;
                    uint priority = 1;
            };

            using Ptr  = std::shared_ptr<Thread>;
            using List = std::vector<Ptr>;

            enum class State { NotStarted, Running, Finished };

            Thread(Task::Ptr task, StopToken token);
            ~Thread();

            Thread(const Thread&)            = delete;
            Thread& operator=(const Thread&) = delete;

            Thread(Thread&&)                 = delete;
            Thread& operator=(Thread&&)      = delete;

            bool start(uint coremask = 0b1111, uint priority = 1);
            void join();

        private:

            Atomic<State> m_state_;
            Task::Ptr     m_task_;
            TaskHandle_t  m_handle_;
            StopToken     m_token_;
            Semaphore     m_joiner_;
    };

    class Executor {
        public:

            using Ptr   = std::unique_ptr<Executor>;

            ~Executor() = default;

            static const Ptr& get() {
                static const Ptr executor = Ptr(new Executor());
                return executor;
            }

            void launch();
            void cancel();
            void enqueue(Task::Ptr task, uint coremask = 0b1111, uint priority = 1);

        private:

            Executor() = default;

            StopSource   m_stopsrc_;
            Thread::List m_threads_;
    };

} // namespace corekit::platform