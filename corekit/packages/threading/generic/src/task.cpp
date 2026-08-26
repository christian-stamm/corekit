#include "corekit/task.hpp"

#include "corekit/assert.hpp"

namespace corekit {

    Task::Task() : m_state(State::READY) {}

    Result<void> Task::exec(StopToken token) {
        State expected = State::READY;

        if (!m_state.compare_exchange(expected, State::RUNNING)) {
            return RuntimeError("Task is already running or completed");
        }

        Result<void> result;

        if (result) {
            result = on_enter(token);

            if (result) {
                result = on_run(token);

                if (result) {
                    result = on_leave(token);
                }
            }
        }

        m_state.store(State::TERMINATED);
        return result;
    }

}  // namespace corekit