#pragma once

#include <optional>
#include <string>

namespace corekit {
    namespace utils {

        class Watch {
           public:
            Watch(const std::optional<float> timeout = std::nullopt,
                  bool                       trigger = true);

            void reset(bool trigger = true) const;
            bool start() const;
            bool stop() const;

            void  block() const;
            bool  expired() const;
            float remaining() const;
            float elapsed() const;

            std::string  represent() const;
            static float runtime();

           private:
            mutable std::optional<float> timeout;
            mutable std::optional<float> t0, t1;
        };

    };  // namespace utils
};      // namespace corekit
