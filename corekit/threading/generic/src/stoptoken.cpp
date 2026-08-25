#include "corekit/stoptoken.hpp"

namespace corekit {

    static_assert(StopTokenLike<StopToken>);
    static_assert(StopSourceLike<StopSource, StopToken>);

};  // namespace corekit