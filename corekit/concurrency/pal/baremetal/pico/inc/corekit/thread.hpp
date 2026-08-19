#pragma once
#include <memory>

#include "corekit/atomic.hpp"
#include "corekit/concepts.hpp"
#include "corekit/task.hpp"
#include "pico/multicore.h"

namespace corekit {

    class PicoThread {
       public:
        using Ptr = std::shared_ptr<PicoThread>;

        PicoThread(const Task::Ptr& task) : task_(task) {}

        ~PicoThread() {
            request_stop();

            if (joinable()) {
                join();
            }
        }

        void run() {
            task_->exec(stopsrc_.get_token());
        }

        void request_stop() {
            stopsrc_.request_stop();
        }

        void join();

        bool joinable() const {
            return stopsrc_.stop_possible();
        }

       private:
        StopSource stopsrc_;
        Task::Ptr  task_;
    };

    using Thread = PicoThread;

}  // namespace corekit
