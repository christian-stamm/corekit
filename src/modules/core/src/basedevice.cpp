#include "corekit/device/basedevice.hpp"

namespace corekit {

    BaseDevice::BaseDevice(const Name& name) : name(name), loaded(false) {}

    BaseDevice::~BaseDevice() {
        unload();
    }

    bool BaseDevice::load() {
        bool expected = false;
        bool desired  = true;
        // Transition from not-loaded -> loaded once.
        // Only the thread that successfully flips the flag runs prepare().
        if (loaded.compare_exchange(expected, desired)) {
            try {
                watch.reset(true);
                return onLoad();
            } catch (...) {
                loaded.store(false);
                throw;  // Preserve original exception details.
            }
        }

        return false;
    }

    bool BaseDevice::unload() {
        bool expected = true;
        bool desired  = false;
        // Transition from loaded -> not-loaded once.
        // The thread that wins runs cleanup().
        if (loaded.compare_exchange(expected, desired)) {
            try {
                watch.stop();
                return onUnload();
            } catch (...) {
                loaded.store(true);
                throw;  // Preserve original exception details.
            }
        }

        return false;
    }

    bool BaseDevice::reload() {
        bool success = isLoaded();

        if (isLoaded()) {
            success &= unload();
        }

        assert(!isLoaded());
        success &= load();
        return success;
    }

    bool BaseDevice::isLoaded() const {
        return loaded.load();
    }

    double BaseDevice::uptime() const {
        return watch.elapsed();
    }
};  // namespace corekit
