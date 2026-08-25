#include "corekit/conditionvariable.hpp"

#include <mutex>

#include "corekit/mutex.hpp"

namespace corekit {

    static_assert(
        ConditionVariableLike<ConditionVariable, std::unique_lock<Mutex>>);

};  // namespace corekit