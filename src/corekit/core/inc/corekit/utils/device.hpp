#pragma once

#include <atomic>

#include "corekit/utils/logger.hpp"
#include "corekit/utils/watch.hpp"

namespace corekit {
    namespace utils {

        using namespace corekit::types;

        class Device {
           public:
            using Ptr = std::shared_ptr<Device>;

            Device(const Name& name);
            Device(const Device& other);
            Device& operator=(const Device& other);

            virtual ~Device();

            bool   load();
            bool   unload();
            bool   reload();
            bool   isLoaded() const;
            double uptime() const;

            Name   name;
            Logger logger;

           protected:
            virtual bool prepare() {
                return true;
            };
            virtual bool cleanup() {
                return true;
            };

           private:
            std::atomic_bool loaded;
            Watch            watch;
        };

    };  // namespace utils
};      // namespace corekit
