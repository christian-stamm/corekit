#pragma once
#include <memory>
#include <mutex>

namespace corekit::runtime {

    class Mutex : public std::mutex {
        using Ptr = std::shared_ptr<Mutex>;
        using std::mutex::mutex;
    };

};  // namespace corekit::runtime