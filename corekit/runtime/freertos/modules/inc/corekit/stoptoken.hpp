#pragma once

#include <memory>

#include "corekit/atomic.hpp"

namespace corekit {

    using StopState = Atomic<bool>;

    class StopToken {
       public:
        explicit StopToken(const StopState::Ptr& state);

        bool stop_requested() const;
        bool stop_possible() const;

       private:
        StopState::Ptr m_state;
    };

    using StopToken = StopToken;

    class StopSource {
       public:
        using Ptr = std::shared_ptr<StopSource>;

        StopSource();

        bool stop_requested() const;
        bool stop_possible() const;
        bool request_stop();

        StopToken get_token() const;

       private:
        StopState::Ptr  m_state;
        const StopToken m_token;
    };

    using StopSource = StopSource;

}  // namespace corekit