#pragma once

#include "corekit/atomic.hpp"
#include "corekit/mutex.hpp"
#include "corekit/smphr.hpp"
#include "corekit/thread.hpp"

namespace corekit {

    template <typename T>
    using Atomic     = PosixAtomic<T>;
    using Mutex      = PosixMutex;
    using Semaphore  = PosixSemaphore;
    using StopToken  = PosixStopToken;
    using StopSource = PosixStopSource;
    using Thread     = PosixThread;

}  // namespace corekit