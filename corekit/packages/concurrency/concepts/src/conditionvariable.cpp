#include "corekit/conditionvariable.hpp"

#include <functional>

#include "corekit/mutex.hpp"

namespace corekit {

    static_assert(
        ConditionVariableLike<ConditionVariable, Mutex, std::function<bool()>>);

};  // namespace corekit