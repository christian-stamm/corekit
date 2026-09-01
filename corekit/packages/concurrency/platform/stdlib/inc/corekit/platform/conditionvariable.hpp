#pragma once
#include <condition_variable>
#include <memory>

#include "corekit/mutex.hpp"

namespace corekit::platform {

    class ConditionVariable : public std::condition_variable_any {
       public:
        using Ptr = std::shared_ptr<ConditionVariable>;
        using std::condition_variable_any::condition_variable_any;

        ConditionVariable(uint){};
    };

};  // namespace corekit::platform