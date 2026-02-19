#include "corekit/utils/watch.hpp"

#include <chrono>
#include <format>
#include <iostream>
#include <thread>

namespace corekit {
    namespace utils {

        using time = std::chrono::high_resolution_clock;

        Watch::Watch(const std::optional<float> timeout, bool trigger)
            : timeout(timeout) {
            this->reset(trigger);
        }

        void Watch::reset(bool trigger) const {
            t0.reset();
            t1.reset();

            if (trigger) {
                start();
            }
        }

        bool Watch::start() const {
            if (!t0.has_value() && !t1.has_value()) {
                t0 = runtime();
            }

            return t0.has_value();
        }

        bool Watch::stop() const {
            if (t0.has_value() && !t1.has_value()) {
                t1 = runtime();
            }

            return t1.has_value();
        }

        void Watch::block() const {
            while (!expired()) {
                std::this_thread::sleep_for(
                    std::chrono::duration<double>(0.95 * remaining()));
            };
        }

        bool Watch::expired() const {
            return remaining() <= 0.0f;
        }

        float Watch::remaining() const {
            return std::max<float>(timeout.value_or(0.0f) - elapsed(), 0.0f);
        }

        float Watch::elapsed() const {
            const float upper = t1.value_or(runtime());
            const float lower = t0.value_or(upper);
            return std::max<float>(upper - lower, 0.0f);
        }

        float Watch::tick() const {
            const float dt = elapsed();
            this->reset(true);
            return dt;
        }

        std::string Watch::represent() const {
            return std::format("Watch(Elapsed={:.6f}s, Remaining={:.6f}s)",
                               elapsed(),
                               remaining());
        }

        float Watch::runtime() {
            static const auto ref = time::now();

            const auto t = time::now();
            const auto n =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t - ref);
            return double(1e-9 * double(n.count()));
        }

    };  // namespace utils
};      // namespace corekit
