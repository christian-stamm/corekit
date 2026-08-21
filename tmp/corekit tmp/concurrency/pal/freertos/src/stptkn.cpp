#include <stdexcept>

#include "corekit/stoptoken.hpp"

namespace corekit {

    FreeRTOSStopToken::FreeRTOSStopToken(const StopState::Ptr& state)
        : m_state(state) {
        if (!m_state) {
            throw std::invalid_argument("StopState pointer cannot be null");
        }
    }

    bool FreeRTOSStopToken::stop_requested() const {
        return m_state && m_state->load();
    }

    bool FreeRTOSStopToken::stop_possible() const {
        return m_state && !m_state->load();
    }

    FreeRTOSStopSource::FreeRTOSStopSource()
        : m_state(std::make_shared<StopState>(false))
        , m_token(m_state) {}

    bool FreeRTOSStopSource::stop_requested() const {
        return m_token.stop_requested();
    }

    bool FreeRTOSStopSource::stop_possible() const {
        return m_token.stop_possible();
    }

    bool FreeRTOSStopSource::request_stop() {
        if (stop_possible()) {
            m_state->store(true);
            return true;
        }

        return false;
    }

    StopToken FreeRTOSStopSource::get_token() const {
        return StopToken(m_state);
    }

}  // namespace corekit