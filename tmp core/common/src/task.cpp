#include "corekit/task.hpp"

#include <array>
#include <exception>
#include <stdexcept>

namespace corekit {

    Task::Task() : m_state(State::READY) {}

    Task::Result Task::exec(StopToken token) {
        State expected = State::READY;

        if (!m_state.compare_exchange_strong(expected, State::RUNNING)) {
            return std::unexpected(std::make_exception_ptr(
                std::runtime_error("Task already launched")));
        }

        using Hook = Result (Task::*)(StopToken);

        constexpr std::array<Hook, 3> pipe = {
            &Task::on_enter,
            &Task::on_run,
            &Task::on_leave,
        };

        try {
            for (auto hook : pipe) {
                auto result = (this->*hook)(token);

                if (!result) {
                    m_state.store(State::FAILED);
                    return result;
                }
            }
        } catch (...) {
            const auto &eptr = std::current_exception();

            m_state.store(State::CRASHED);
            on_exception(eptr);

            return std::unexpected(eptr);
        }

        m_state.store(State::COMPLETED);
        return {};
    }

}  // namespace corekit