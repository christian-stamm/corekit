#include "corekit/conditionvariable.hpp"

#include <functional>
#include <mutex>

namespace corekit {

    static_assert(
        ConditionVariableLike<ConditionVariable, Mutex, std::function<bool()>>);

};  // namespace corekit