#include "corekit/basedevice.hpp"

#include "corekit/check.hpp"

namespace corekit {

    BaseDevice::BaseDevice(const std::string& name)
        : name(name)
        , loaded(false) {}

    BaseDevice::~BaseDevice() {
        unload();
    }

    bool BaseDevice::load() {
        bool expected = false;
        bool desired  = true;

        // Transition from not-loaded -> loaded once.
        // Only the thread that successfully flips the flag runs prepare().

        if (loaded.compare_exchange(expected, desired)) {
            watch.reset(true);

            if (!on_load()) {
                loaded.store(false);
                Error::stack.push(
                    RuntimeError("Failed to load device: " + name));
            }
        }

        return is_loaded();
    }

    bool BaseDevice::unload() {
        bool expected = true;
        bool desired  = false;
        // Transition from loaded -> not-loaded once.
        // The thread that wins runs cleanup().
        if (loaded.compare_exchange(expected, desired)) {
            if (!on_unload()) {
                loaded.store(true);
                Error::stack.push(
                    RuntimeError("Failed to unload device: " + name));
            }
        }

        return !is_loaded();
    }

    bool BaseDevice::reload() {
        bool success = is_loaded();

        if (is_loaded()) {
            success &= unload();
        }

        success &= load();
        return success;
    }

    bool BaseDevice::is_loaded() const {
        return loaded.load();
    }

    double BaseDevice::uptime() const {
        return watch.elapsed();
    }
};  // namespace corekit
