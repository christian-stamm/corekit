#include "corekit/basedevice.hpp"

#include "corekit/check.hpp"

namespace corekit {

    BaseDevice::BaseDevice(const std::string& name)
        : name(name)
        , loaded(false) { }

    BaseDevice::~BaseDevice() { unload(); }

    bool BaseDevice::load() {
        bool expected = false;
        bool desired  = true;

        // Transition from not-loaded -> loaded once.
        // Only the thread that successfully flips the flag runs prepare().

        if (loaded.compare_exchange(expected, desired)) {
            watch.reset(true);

            core::check(on_load(), RuntimeError("Failed to load device: " + name));
        }

        return is_loaded() && core::ok();
    }

    bool BaseDevice::unload() {
        bool expected = true;
        bool desired  = false;
        // Transition from loaded -> not-loaded once.
        // The thread that wins runs cleanup().
        if (loaded.compare_exchange(expected, desired)) {
            core::check(on_unload(), RuntimeError("Failed to unload device: " + name));
        }

        return !is_loaded() && core::ok();
    }

    bool BaseDevice::reload() {
        if (is_loaded()) { core::check(unload(), RuntimeError("Failed to unload device: " + name)); }
        return core::check(load(), RuntimeError("Failed to load device: " + name));
    }

    bool BaseDevice::is_loaded() const { return loaded.load(); }

    double BaseDevice::uptime() const { return watch.elapsed(); }
}; // namespace corekit
