#pragma once
#include <exception>

#include "corekit/atomic.hpp"
#include "corekit/concepts.hpp"
#include "corekit/stoptoken.hpp"

namespace corekit {

    class Task {
       public:
        using Ptr = std::shared_ptr<Task>;

        Task() : launched_(false), crashed_(false), completed_(false) {}
        Task(const Task&)             = delete;
        Task(const Task&&)            = delete;
        Task& operator=(const Task&)  = delete;
        Task& operator=(const Task&&) = delete;

        virtual bool exec(const StopToken& token) final {
            launched_   = true;
            completed_  = false;
            crashed_    = false;
            bool result = true;

            try {
                result = on_enter(token) && on_run(token) && on_leave(token);
            } catch (...) {
                on_error(std::current_exception());
                crashed_ = true;
            }

            completed_ = true;
            return result;
        }

        bool is_launched() const {
            return launched_;
        }
        bool is_crashed() const {
            return crashed_;
        }
        bool is_running() const {
            return (launched_ && !completed_) && !crashed_;
        }
        bool is_completed() const {
            return completed_;
        }

       protected:
        virtual bool on_enter(const StopToken& token) {
            return true;
        }
        virtual bool on_run(const StopToken& token) = 0;
        virtual bool on_leave(const StopToken& token) {
            return true;
        }
        virtual void on_error(std::exception_ptr eptr) {}

       private:
        bool crashed_;
        bool launched_;
        bool completed_;
    };

}  // namespace corekit