#pragma once
#include <memory>
#include <thread>

#include "corekit/task.hpp"

namespace corekit {

    class StdlibThread : public std::jthread {
       public:
        using Ptr = std::shared_ptr<StdlibThread>;

        StdlibThread(Task::Ptr task);

       private:
        using std::jthread::jthread;
        Task::Ptr task_;
    };

    using Thread = StdlibThread;

}  // namespace corekit