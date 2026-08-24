#include "corekit/watch.hpp"

#include <format>

#include "corekit/time.hpp"

namespace corekit {

    Watch::Watch(const Timeout timeout, bool trigger) : timeout(timeout) {
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
            t0 = Time::now();
        }

        return t0.has_value();
    }

    bool Watch::stop() const {
        if (t0.has_value() && !t1.has_value()) {
            t1 = Time::now();
        }

        return t1.has_value();
    }

    void Watch::block() const {
        while (!expired()) {
            Time::sleep(0.5 * remaining());
        };
    }

    bool Watch::expired() const {
        return remaining() <= 0.0f;
    }

    double Watch::remaining() const {
        return std::max<double>(timeout.value_or(0.0) - elapsed(), 0.0);
    }

    double Watch::elapsed() const {
        const double upper = t1.value_or(Time::now());
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

};  // namespace corekit