#pragma once
#include <memory>

namespace corekit {

    class PicoThread {
       public:
        using Ptr = std::shared_ptr<PicoThread>;

        void run();
        void join();
        void detach();
        bool joinable() const;

       private:
    };

}  // namespace corekit
