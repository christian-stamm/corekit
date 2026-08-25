#include "corekit/stoptoken.hpp"

namespace corekit {

    StopToken::StopToken(const StopState::Ptr& state) : m_state(state) {
        if (!m_state) {
            throw std::invalid_argument("StopState pointer cannot be null");
        }
    }

    bool StopToken::stop_requested() const {
        return m_state && m_state->load();
    }

    bool StopToken::stop_possible() const {
        return m_state && !m_state->load();
    }

    StopSource::StopSource()
        : m_state(std::make_shared<StopState>(false))
        , m_token(m_state) {}

    bool StopSource::stop_requested() const {
        return m_token.stop_requested();
    }

    bool StopSource::stop_possible() const {
        return m_token.stop_possible();
    }

    bool StopSource::request_stop() {
        if (stop_possible()) {
            m_state->store(true);
            return true;
        }

        return false;
    }

    StopToken StopSource::get_token() const {
        return StopToken(m_state);
    }

}  // namespace corekit