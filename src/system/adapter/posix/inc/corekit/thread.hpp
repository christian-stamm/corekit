#pragma once
#include <memory>
#include <thread>
namespace corekit {

    class PosixThread {
       public:
        using Ptr = std::shared_ptr<PosixThread>;

        void run();
        void join();
        void detach();
        bool joinable() const;

       private:
        std::thread thread_;
    };

}  // namespace corekit
