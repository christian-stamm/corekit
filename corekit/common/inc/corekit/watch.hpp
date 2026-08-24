#pragma once

#include <optional>
#include <string>

namespace corekit {

    using Timeout   = std::optional<double>;
    using Timestamp = std::optional<double>;

    class Watch {
       public:
        Watch(const Timeout timeout = std::nullopt, bool trigger = true);

        void reset(bool trigger = true) const;
        bool start() const;
        bool stop() const;

        void   block() const;
        bool   expired() const;
        double remaining() const;
        double elapsed() const;
        double tick() const;

        std::string represent() const;

       private:
        mutable Timeout   timeout;
        mutable Timestamp t0, t1;
    };

}  // namespace corekit
