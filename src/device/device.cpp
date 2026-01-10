#include "corekit/device/device.hpp"

#include <string>

namespace corekit {
    namespace device {

        Device::Device(const std::string& name)
            : name(name)
            , logger(name)
            , loaded(false) {}

        Device::~Device() {
            // Destructor must not throw. Ensure we attempt to unload but
            // swallow any exceptions from cleanup.
            try {
                unload();
            } catch (...) {
                logger(Level::ERROR)
                    << ("Exception thrown during device cleanup in destructor");
            }
        }

        bool Device::load() {
            bool expected = false;
            bool desired  = true;
            // Transition from not-loaded -> loaded once. Only the thread that
            // successfully flips the flag runs prepare().
            if (loaded.compare_exchange_strong(expected, desired)) {
                try {
                    return prepare();
                } catch (...) {
                    // If prepare throws, reset loaded flag and rethrow.
                    loaded.store(!desired);
                    throw;
                }
            }

            return true;
        }

        bool Device::unload() {
            bool expected = true;
            bool desired  = false;
            // Transition from loaded -> not-loaded once. The thread that wins
            // runs cleanup().
            if (loaded.compare_exchange_strong(expected, desired)) {
                try {
                    return cleanup();
                } catch (...) {
                    // If cleanup throws, keep loaded flag and rethrow.
                    loaded.store(!desired);
                    throw;
                }
            }

            return true;
        }

        bool Device::reload() {
            return unload() && load();
        }

        bool Device::isLoaded() const {
            return loaded.load();
        }

    };  // namespace device
};  // namespace corekit
