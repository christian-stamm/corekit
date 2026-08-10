#pragma once
#include <thread>

namespace corekit {

 template<typename Callable>
    class PosixThread {

        public:

            explicit PosixThread(Callable&& callable)
                : callable_(std::forward<Callable>(callable))
            {}


            void run()
            {
                thread_ = std::thread(std::move(callable_));
            }

            void join()
            {
                if (this->joinable())
                    thread_.join();
            }

            bool joinable() const {
                return thread_.joinable();
            }

        private:
            
            Callable callable_;
            std::thread thread_;
    };

template<typename Callable>
using Thread = PosixThread<Callable>;

}  // namespace corekit
