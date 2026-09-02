#include "corekit/stack.hpp"

namespace corekit {

    static_assert(StackLike<Stack<bool>, bool>);

};  // namespace corekit