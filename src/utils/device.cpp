#include <exception>
#include <stdexcept>

#include "corekit/core.hpp"

namespace corekit {
    namespace utils {

        Device::Device(const Name& name)
            : name(name)
            , logger(name)
            , loaded(false) {}

        Device::Device(const Device& other)
            : name(other.name)
            , logger(other.logger)
            , loaded(false) {}

        Device::~Device() {
            unload();
        }

        Device& Device::operator=(const Device& other) {
            if (this != &other) {
                this->unload();
                name   = other.name;
                logger = other.logger;
                loaded.store(false);
            }

            return *this;
        }

        bool Device::load() {
            bool expected = false;
            bool desired  = true;
            // Transition from not-loaded -> loaded once.
            // Only the thread that successfully flips the flag runs prepare().
            if (loaded.compare_exchange_strong(expected, desired)) {
                try {
                    watch.reset(true);
                    return prepare();
                } catch (const std::exception& e) {
                    loaded.store(!desired);
                    logger(Level::ERROR) << "Device load failed: " << e.what();
                    throw e;
                }
            }

            return true;
        }

        bool Device::unload() {
            bool expected = true;
            bool desired  = false;
            // Transition from loaded -> not-loaded once.
            // The thread that wins runs cleanup().
            if (loaded.compare_exchange_strong(expected, desired)) {
                try {
                    watch.stop();
                    return cleanup();
                } catch (const std::exception& e) {
                    loaded.store(!desired);
                    logger(Level::ERROR)
                        << "Device unload failed: " << e.what();
                    throw e;
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

        double Device::uptime() const {
            return watch.elapsed();
        }

    };  // namespace utils
};  // namespace corekit
