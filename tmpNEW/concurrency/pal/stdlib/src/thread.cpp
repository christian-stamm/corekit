#include "corekit/thread.hpp"

#include <utility>

namespace corekit {

    StdlibThread::StdlibThread(Task::Ptr task)
        : std::jthread(
              [task = std::move(task)](StopToken token) { task->exec(token); })
        , task_(std::move(task)) {}

}  // namespace corekit