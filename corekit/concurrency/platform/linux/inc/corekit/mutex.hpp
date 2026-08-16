#pragma once
#include <memory>
#include <mutex>

namespace corekit {

    class PosixMutex : public std::mutex {
       public:
        using Ptr = std::shared_ptr<PosixMutex>;
        using std::mutex::mutex;
    };

}  // namespace corekit
