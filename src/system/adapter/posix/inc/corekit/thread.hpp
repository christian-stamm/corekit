#pragma once
#include <memory>
#include <thread>
namespace corekit {

    class PosixThread {
       public:
        using Ptr = std::shared_ptr<PosixThread>;

        void run() {
            // thread_ = std::thread(std::move(callable_));
        }

        void join() {
            if (this->joinable())
                thread_.join();
        }

        void detach() {
            if (this->joinable())
                thread_.detach();
        }

        bool joinable() const {
            return thread_.joinable();
        }

       private:
        std::thread thread_;
    };

}  // namespace corekit
