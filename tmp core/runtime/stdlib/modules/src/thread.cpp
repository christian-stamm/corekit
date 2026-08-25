#include "corekit/thread.hpp"

#include <utility>

namespace corekit {

    Thread::Thread(Task::Ptr task)
        : std::jthread([task = std::forward<Task::Ptr>(task)](StopToken token) {
            task->exec(token);
        })
        , task_(std::forward<Task::Ptr>(task)) {}

}  // namespace corekit