#pragma once

#include <memory>

#include "corekit/atomic.hpp"

namespace corekit::platform {

    using StopState = Atomic<bool>;

    class StopToken {
        friend class StopSource;

       public:
        explicit StopToken(const StopState::Ptr& state = nullptr);

        bool stop_requested() const;
        bool stop_possible() const;

       private:
        StopState::Ptr m_state;
    };

    class StopSource : public StopToken {
       public:
        using Ptr = std::shared_ptr<StopSource>;

        StopSource();

        bool      request_stop();
        StopToken get_token() const;
    };

}  // namespace corekit::platform