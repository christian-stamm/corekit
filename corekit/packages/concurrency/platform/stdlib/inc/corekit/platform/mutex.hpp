#pragma once
#include <memory>
#include <mutex>

namespace corekit::platform {

    class Mutex : public std::mutex {
        using Ptr = std::shared_ptr<Mutex>;
        using std::mutex::mutex;
    };

};  // namespace corekit::platform