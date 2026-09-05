#include "corekit/basedevice.hpp"

namespace corekit {

    BaseDevice::BaseDevice(const std::string& name)
        : name(name)
        , loaded(false) {}

    BaseDevice::~BaseDevice() {
        if (is_loaded()) {
            unload();
        }
    }

    VoidResult BaseDevice::load() {
        bool expected = false;
        bool desired  = true;
        // Transition from not-loaded -> loaded once.
        // Only the thread that successfully flips the flag runs prepare().
        if (loaded.compare_exchange(expected, desired)) {
            watch.reset(true);
            return on_load();
        }

        return VoidResult();
    }

    VoidResult BaseDevice::unload() {
        bool expected = true;
        bool desired  = false;
        // Transition from loaded -> not-loaded once.
        // The thread that wins runs cleanup().
        if (loaded.compare_exchange(expected, desired)) {
            watch.stop();
            return on_unload();
        }

        return VoidResult();
    }

    VoidResult BaseDevice::reload() {
        if (!(unload() && load())) {
            return RuntimeError("Failed to reload device: " + name + ".");
        }

        return VoidResult();
    }

    bool BaseDevice::is_loaded() const {
        return loaded.load();
    }

    double BaseDevice::uptime() const {
        return watch.elapsed();
    }
};  // namespace corekit
