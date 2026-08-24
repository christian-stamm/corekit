#pragma once

#include <optional>
#include <string>

namespace corekit {

    class Watch {
       public:
        Watch(const std::optional<double> timeout = std::nullopt,
              bool                        trigger = true);

        void reset(bool trigger = true) const;
        bool start() const;
        bool stop() const;

        void   block() const;
        bool   expired() const;
        double remaining() const;
        double elapsed() const;
        double tick() const;

        std::string   represent() const;
        static double runtime();

       private:
        mutable std::optional<double> timeout;
        mutable std::optional<double> t0, t1;
    };

}  // namespace corekit
