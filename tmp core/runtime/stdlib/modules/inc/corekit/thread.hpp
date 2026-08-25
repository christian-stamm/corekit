#pragma once
#include <memory>
#include <thread>

#include "corekit/task.hpp"

namespace corekit {

    class Thread : public std::jthread {
       public:
        using Ptr = std::shared_ptr<Thread>;
        Thread(Task::Ptr task);

       private:
        using std::jthread::jthread;
        Task::Ptr task_;
    };

}  // namespace corekit