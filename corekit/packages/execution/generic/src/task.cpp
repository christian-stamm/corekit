#include "corekit/task.hpp"

namespace corekit {

    Task::Task(const std::string& name = "task")
        : name(name)
        , logger(name)
        , m_state(State::READY) {}

    VoidResult Task::exec(StopToken token) noexcept {
        State expected = State::READY;

        if (!m_state.compare_exchange(expected, State::RUNNING)) {
            return RuntimeError("Task is already running or completed");
        }

        if (!on_enter(token)) {
            m_state.store(State::ERROR);
            return RuntimeError("Task on_enter failed");
        }

        if (!on_run(token)) {
            m_state.store(State::ERROR);
            return RuntimeError("Task on_run failed");
        }

        if (!on_leave(token)) {
            m_state.store(State::ERROR);
            return RuntimeError("Task on_leave failed");
        }

        m_state.store(State::TERMINATED);
        return VoidResult();
    }

}  // namespace corekit