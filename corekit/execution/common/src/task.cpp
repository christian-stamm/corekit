#include "corekit/task.hpp"

#include "corekit/check.hpp"

namespace corekit {

    Task::Task(const std::string& name = "task")
        : name(name)
        , logger(name)
        , state_(State::READY) { }

    bool Task::exec(StopToken token) noexcept {
        State expected = State::READY;

        core::check(
            state_.compare_exchange(expected, State::RUNNING),
            RuntimeError("Task is already running or completed"));

        core::check(on_init(token), RuntimeError("Task on_init failed"));
        core::check(on_exec(token), RuntimeError("Task on_exec failed"));
        core::check(on_exit(token), RuntimeError("Task on_exit failed"));

        if (!core::ok()) {
            state_.store(State::ERROR);
            return false;
        } else {
            state_.store(State::TERMINATED);
            return true;
        }
    }

} // namespace corekit