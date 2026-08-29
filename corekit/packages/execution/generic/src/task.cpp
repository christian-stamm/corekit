#include "corekit/task.hpp"

namespace corekit {

    Task::Task() : m_state(State::READY) {}

    VoidResult Task::exec(StopToken token) noexcept {
        State expected = State::READY;

        if (!m_state.compare_exchange(expected, State::RUNNING)) {
            return RuntimeError("Task is already running or completed");
        }

        VoidResult result;

        try {
            if (result) {
                result = on_enter(token);

                if (result) {
                    result = on_run(token);

                    if (result) {
                        result = on_leave(token);
                    }
                }
            }
        } catch (const std::exception& e) {
            result = RuntimeError(e.what());
        }

        m_state.store(State::TERMINATED);
        return result;
    }

}  // namespace corekit