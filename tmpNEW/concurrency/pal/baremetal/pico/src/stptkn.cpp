#include <stdexcept>

#include "corekit/stoptoken.hpp"

namespace corekit {

    PicoStopToken::PicoStopToken(const StopState::Ptr& state) : m_state(state) {
        if (!m_state) {
            throw std::invalid_argument("StopState pointer cannot be null");
        }
    }

    bool PicoStopToken::stop_requested() const {
        return m_state && m_state->load();
    }

    bool PicoStopToken::stop_possible() const {
        return m_state && !m_state->load();
    }

    PicoStopSource::PicoStopSource()
        : m_state(std::make_shared<StopState>(false))
        , m_token(m_state) {}

    bool PicoStopSource::stop_requested() const {
        return m_token.stop_requested();
    }

    bool PicoStopSource::stop_possible() const {
        return m_token.stop_possible();
    }

    bool PicoStopSource::request_stop() {
        if (stop_possible()) {
            m_state->store(true);
            return true;
        }

        return false;
    }

    StopToken PicoStopSource::get_token() const {
        return StopToken(m_state);
    }

}  // namespace corekit