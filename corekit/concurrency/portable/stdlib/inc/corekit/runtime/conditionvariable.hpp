#pragma once
#include <condition_variable>
#include <memory>

namespace corekit::runtime {

    class ConditionVariable : public std::condition_variable_any {
        using Ptr = std::shared_ptr<ConditionVariable>;
        using std::condition_variable_any::condition_variable_any;
    };

};  // namespace corekit::runtime