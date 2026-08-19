#pragma once
#include <memory>

#include "corekit/atomic.hpp"

namespace corekit {

    using StopState = Atomic<bool>;

    class PicoStopToken {
       public:
        PicoStopToken(const StopState::Ptr& state) : m_state(state) {
            if (!m_state) {
                throw std::invalid_argument("StopState pointer cannot be null");
            }
        }

        bool stop_requested() const {
            return m_state && m_state->load();
        }

        bool stop_possible() const {
            return m_state && !m_state->load();
        }

       private:
        StopState::Ptr m_state;
    };

    using StopToken = PicoStopToken;

    class PicoStopSource {
       public:
        using Ptr = std::shared_ptr<PicoStopSource>;

        PicoStopSource()
            : m_state(std::make_shared<StopState>(false))
            , m_token(m_state) {}

        bool stop_requested() const {
            return m_token.stop_requested();
        }

        bool stop_possible() const {
            return m_token.stop_possible();
        }

        bool request_stop() {
            if (stop_possible()) {
                m_state->store(true);
                return true;
            }

            return false;
        }

        StopToken get_token() const {
            return StopToken(m_state);
        }

       private:
        static void deamon() {}

        StopState::Ptr  m_state;
        const StopToken m_token;
    };

    using StopSource = PicoStopSource;

}  // namespace corekit
