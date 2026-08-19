#pragma once
#include <memory>
#include <semaphore>

namespace corekit {

    class StdlibSemaphore : public std::counting_semaphore<> {
       public:
        using Ptr = std::shared_ptr<StdlibSemaphore>;
        using std::counting_semaphore<>::counting_semaphore;
    };

    using Semaphore = StdlibSemaphore;

}  // namespace corekit
