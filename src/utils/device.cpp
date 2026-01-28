#include "corekit/core.hpp"

namespace corekit {
    namespace utils {

        Device::Device(const Name& name)
            : name(name)
            , logger(name)
            , loaded(false) {}

        Device::~Device() {
            unload();
        }

        bool Device::load() {
            bool expected = false;
            bool desired  = true;
            // Transition from not-loaded -> loaded once. Only the thread that
            // successfully flips the flag runs prepare().
            if (loaded.compare_exchange_strong(expected, desired)) {
                try {
                    watch.reset(true);
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
                    watch.stop();
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

    };  // namespace utils
};      // namespace corekit
