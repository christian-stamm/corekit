#pragma once

#include <memory>

#include "corekit/stoptoken.hpp"

namespace corekit {

    class Task {
       public:
        using Ptr = std::shared_ptr<Task>;

        Task();
        Task(const Task&)            = delete;
        Task(Task&&)                 = delete;
        Task& operator=(const Task&) = delete;
        Task& operator=(Task&&)      = delete;

        virtual ~Task() = default;

        bool exec(const StopToken& token);

        bool is_launched() const;
        bool is_crashed() const;
        bool is_running() const;
        bool is_completed() const;

       protected:
        virtual bool on_enter(const StopToken& token);
        virtual bool on_leave(const StopToken& token);
        virtual void on_error(std::exception_ptr eptr);
        virtual bool on_run(const StopToken& token) = 0;

       private:
        bool crashed_;
        bool launched_;
        bool completed_;
    };

}  // namespace corekit