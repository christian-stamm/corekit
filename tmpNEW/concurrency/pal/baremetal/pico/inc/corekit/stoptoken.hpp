#pragma once

#include <memory>

#include "corekit/atomic.hpp"

namespace corekit {

    using StopState = Atomic<bool>;

    class PicoStopToken {
       public:
        explicit PicoStopToken(const StopState::Ptr& state);

        bool stop_requested() const;
        bool stop_possible() const;

       private:
        StopState::Ptr m_state;
    };

    using StopToken = PicoStopToken;

    class PicoStopSource {
       public:
        using Ptr = std::shared_ptr<PicoStopSource>;

        PicoStopSource();

        bool stop_requested() const;
        bool stop_possible() const;
        bool request_stop();

        StopToken get_token() const;

       private:
        StopState::Ptr  m_state;
        const StopToken m_token;
    };

    using StopSource = PicoStopSource;

}  // namespace corekit