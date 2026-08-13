#pragma once
#include <thread>

#include "corekit/iface/thread.hpp"
#include "corekit/task.hpp"

namespace corekit {

    class PosixThread : public IThread {
       public:
        explicit PosixThread(const ITask& task) : task_(task) {}

        void run() {
            thread_ = std::thread(std::move(callable_));
        }

        void join() {
            if (this->joinable())
                thread_.join();
        }

        bool joinable() const {
            return thread_.joinable();
        }

       private:
        ITask       task_;
        std::thread thread_;
    };

    using Thread = PosixThread;

}  // namespace corekit
