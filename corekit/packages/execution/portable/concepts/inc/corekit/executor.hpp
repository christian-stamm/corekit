#pragma once

#include <concepts>

#include "corekit/platform/executor.hpp"
#include "corekit/task.hpp"

namespace corekit {

    using Executor = platform::Executor;

    template <typename T>
    concept ExecutorLike =  //
        requires(T executor, Task::Ptr task) {
            { executor.enqueue(task) } -> std::convertible_to<void>;
            { executor.cancel() } -> std::convertible_to<void>;
        };

}  // namespace corekit