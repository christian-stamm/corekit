#pragma once

#include <memory>

#include "corekit/atomic.hpp"

namespace corekit {

    using StopState = Atomic<bool>;

    class FreeRTOSStopToken {
       public:
        explicit FreeRTOSStopToken(const StopState::Ptr& state);

        bool stop_requested() const;
        bool stop_possible() const;

       private:
        StopState::Ptr m_state;
    };

    using StopToken = FreeRTOSStopToken;

    class FreeRTOSStopSource {
       public:
        using Ptr = std::shared_ptr<FreeRTOSStopSource>;

        FreeRTOSStopSource();

        bool stop_requested() const;
        bool stop_possible() const;
        bool request_stop();

        StopToken get_token() const;

       private:
        StopState::Ptr  m_state;
        const StopToken m_token;
    };

    using StopSource = FreeRTOSStopSource;

}  // namespace corekit