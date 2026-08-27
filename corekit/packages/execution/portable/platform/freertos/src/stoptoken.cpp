#include "corekit/platform/stoptoken.hpp"

namespace corekit {

    StopToken::StopToken(const StopState::Ptr& state) : m_state(state) {}

    bool StopToken::stop_requested() const {
        return m_state && m_state->load();
    }

    bool StopToken::stop_possible() const {
        return m_state && !m_state->load();
    }

    StopSource::StopSource() {
        m_state = std::make_shared<StopState>(false);
    }

    bool StopSource::request_stop() {
        if (stop_possible()) {
            m_state->store(true);
            return true;
        }

        return false;
    }

    StopToken StopSource::get_token() const {
        return std::move(StopToken(m_state));
    }

}  // namespace corekit