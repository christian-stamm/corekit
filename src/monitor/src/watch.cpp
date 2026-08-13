#include "corekit/watch.hpp"

#include <chrono>
#include <format>
#include <thread>

namespace corekit {

    Watch::Watch(const std::optional<double> timeout, bool trigger)
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

    double Watch::remaining() const {
        return std::max<double>(timeout.value_or(0.0) - elapsed(), 0.0);
    }

    double Watch::elapsed() const {
        const double upper = t1.value_or(runtime());
        const double lower = t0.value_or(upper);
        return std::max<double>(upper - lower, 0.0);
    }

    double Watch::tick() const {
        const double dt = elapsed();
        this->reset(true);
        return dt;
    }

    std::string Watch::represent() const {
        return std::format("Watch(Elapsed={:.6f}s, Remaining={:.6f}s)",
                           elapsed(),
                           remaining());
    }

    double Watch::runtime() {
        using time = std::chrono::steady_clock;

        static const std::chrono::time_point base    = time::now();
        const auto                           current = time::now();
        const auto                           delta   = current - base;
        const auto                           nanos =
            std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
        return double(1e-9 * double(nanos.count()));
    }

};  // namespace corekit