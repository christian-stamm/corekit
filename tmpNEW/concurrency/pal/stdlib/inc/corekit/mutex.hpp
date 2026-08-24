#pragma once
#include <memory>
#include <mutex>

namespace corekit {

    class StdlibMutex : public std::mutex {
       public:
        using Ptr = std::shared_ptr<StdlibMutex>;
        using std::mutex::mutex;
    };

    using Mutex = StdlibMutex;

}  // namespace corekit
