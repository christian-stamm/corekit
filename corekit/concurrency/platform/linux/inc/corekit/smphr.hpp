#pragma once
#include <memory>
#include <semaphore>

namespace corekit {

    class PosixSemaphore : public std::counting_semaphore<> {
       public:
        using Ptr = std::shared_ptr<PosixSemaphore>;
        using std::counting_semaphore<>::counting_semaphore;
    };

}  // namespace corekit
