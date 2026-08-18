#pragma once
#include <memory>
#include <thread>
#include <stop_token>

namespace corekit {

    using StopToken  = std::stop_token;
    using StopSource = std::stop_source;
    
    class StdlibThread : public std::jthread {
       public:
        using Ptr = std::shared_ptr<StdlibThread>;
        using std::jthread::jthread;
    };

    using Thread = StdlibThread;

}  // namespace corekit
