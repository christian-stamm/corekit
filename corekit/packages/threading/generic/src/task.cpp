#include "corekit/task.hpp"

#include <array>

#include "corekit/assert.hpp"

namespace corekit {

    Task::Task() : m_state(State::READY) {}

    Result<void> Task::exec(StopToken token) {
        State expected = State::READY;

        if (!m_state.compare_exchange(expected, State::RUNNING)) {
            return std::unexpected(
                RuntimeError("Task is already running or completed"));
        }

        using Hook = Result<void> (Task::*)(StopToken);

        constexpr std::array<Hook, 3> pipe = {
            &Task::on_enter,
            &Task::on_run,
            &Task::on_leave,
        };

        for (auto hook : pipe) {
            auto result = (this->*hook)(token);

            if (!result) {
                m_state.store(State::TERMINATED);
                return result;
            }
        }

        m_state.store(State::TERMINATED);
        return Result<void>(std::in_place);
    }

}  // namespace corekit