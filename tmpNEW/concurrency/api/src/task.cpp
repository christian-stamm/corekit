#include "corekit/task.hpp"

#include <exception>

namespace corekit {

    Task::Task() : crashed_(false), launched_(false), completed_(false) {}

    bool Task::exec(const StopToken& token) {
        launched_  = true;
        completed_ = false;
        crashed_   = false;

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

    bool Task::is_launched() const {
        return launched_;
    }

    bool Task::is_crashed() const {
        return crashed_;
    }

    bool Task::is_running() const {
        return (launched_ && !completed_) && !crashed_;
    }

    bool Task::is_completed() const {
        return completed_;
    }

    bool Task::on_enter(const StopToken& token) {
        return true;
    }

    bool Task::on_leave(const StopToken& token) {
        return true;
    }

    void Task::on_error(std::exception_ptr eptr) {}

}  // namespace corekit