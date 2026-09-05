#pragma once

#include <FreeRTOS.h>
#include <task.h>

#include "corekit/semaphore.hpp"
#include "corekit/task.hpp"
#include "corekit/time.hpp"

namespace corekit::platform {

    class Thread {
       public:
        using Ptr = std::shared_ptr<Thread>;

        enum class State { NotStarted, Running, Finished };

        Thread(Task::Ptr task, StopToken token);
        ~Thread();

        Thread(const Thread&)            = delete;
        Thread& operator=(const Thread&) = delete;

        Thread(Thread&&)            = delete;
        Thread& operator=(Thread&&) = delete;

        bool start();
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
        static void launch();
        static void terminate();
    };

}  // namespace corekit::platform