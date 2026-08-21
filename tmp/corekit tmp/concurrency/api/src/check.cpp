// #include "corekit/atomic.hpp"
// #include "corekit/concepts.hpp"
// #include "corekit/mutex.hpp"
// #include "corekit/semaphore.hpp"
// #include "corekit/stoptoken.hpp"
// #include "corekit/time.hpp"

// namespace corekit {

//     static_assert(TimeType<Time>,  //
//                   "Platform does not satisfy the Time concept");

//     static_assert(AtomicType<Atomic<bool>, bool>,
//                   "Platform does not satisfy the Atomic concept");

//     static_assert(MutexType<Mutex>,
//                   "Platform does not satisfy the Mutex concept");

//     static_assert(SemaphoreType<Semaphore>,
//                   "Platform does not satisfy the Semaphore concept");

//     static_assert(StopTokenType<StopToken>,
//                   "Platform does not satisfy the StopToken concept");

//     static_assert(StopSourceType<StopSource, StopToken>,
//                   "Platform does not satisfy the StopSource concept");

// }  // namespace corekit
