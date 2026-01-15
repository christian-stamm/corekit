#pragma once

#include "corekit/device/device.hpp"
#include "corekit/system/concurrency/flow/scheduler.hpp"
#include "corekit/system/diagnostics/logger.hpp"
#include "corekit/system/diagnostics/watch.hpp"
#include "corekit/types.hpp"

namespace corekit {

    namespace system {

        using namespace corekit::types;
        using namespace corekit::device;
        using namespace corekit::system::diagnostics;
        using namespace corekit::system::concurrency;

        Hash getEnv(const Hash& key);

        class Manager : public Device {
           public:
            struct Settings {
                Settings(size_t workers = 4, File::Path dir = RESSOURCE_DIR)
                    : numWorker(workers)
                    , workdir(dir) {}

                size_t     numWorker;
                File::Path workdir;
            };

            Manager(const Settings& settings = Settings());
            Manager& operator<<(const Settings& config);

            void  shutdown();
            bool  ok() const;
            float time() const;

            Scheduler::Ptr scheduler;
            File::Path     workdir;

           protected:
            Watch   clock;
            Killreq killreq;
        };

        inline Manager sys;

    };  // namespace system
};      // namespace corekit
