#include "corekit/concepts.hpp"
#include "corekit/platform.hpp"

namespace corekit {

    static_assert(AtomicType<Atomic<bool>>,
                  "Platform does not satisfy the Atomic concept");

    static_assert(MutexType<Mutex>,
                  "Platform does not satisfy the Mutex concept");

    static_assert(SemaphoreType<Semaphore>,
                  "Platform does not satisfy the Semaphore concept");

    static_assert(ThreadType<Thread>,
                  "Platform does not satisfy the Thread concept");

}  // namespace corekit
