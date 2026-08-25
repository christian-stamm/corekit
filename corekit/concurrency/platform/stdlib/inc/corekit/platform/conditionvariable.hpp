#pragma once
#include <condition_variable>
#include <memory>

namespace corekit::platform {

    class ConditionVariable : public std::condition_variable {
        using Ptr = std::shared_ptr<ConditionVariable>;
        using std::condition_variable::condition_variable;
    };

};  // namespace corekit::platform