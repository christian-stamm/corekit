#pragma once

#include <atomic>

#include "corekit/logging/logger.hpp"

namespace corekit {
    namespace device {

        using namespace corekit::logging;

        class Device {
           public:
            using Ptr = std::shared_ptr<Device>;

            Device(const std::string& name);
            Device(const Device& other) = delete;

            virtual ~Device();

            bool load();
            bool unload();
            bool reload();
            bool isLoaded() const;

            std::string name;
            Logger      logger;

           private:
            virtual bool prepare() {
                return true;
            };
            virtual bool cleanup() {
                return true;
            };

            std::atomic_bool loaded;
        };

    };  // namespace device
};  // namespace corekit
