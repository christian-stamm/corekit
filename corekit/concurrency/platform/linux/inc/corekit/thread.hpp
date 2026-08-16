#pragma once
#include <memory>
#include <thread>
#include <stop_token>

namespace corekit {

    using PosixStopToken = std::stop_token;
    using PosixStopSource = std::stop_source;
    
    class PosixThread : public std::jthread {
       public:
        using Ptr = std::shared_ptr<PosixThread>;
        using std::jthread::jthread;
    };

}  // namespace corekit
