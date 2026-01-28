#pragma once
#include "corekit/system/flow/scheduler.hpp"
#include "corekit/types.hpp"
#include "corekit/utils/device.hpp"
#include "corekit/utils/watch.hpp"

namespace corekit {

    namespace system {

        using namespace corekit::types;
        using namespace corekit::utils;

        Hash getEnv(const Hash& key);

        struct BaseConfig {};

        template <typename Config = BaseConfig>
        class System : public Scheduler {
           public:
            System(Config config    = {},
                   size_t numWorker = 4,
                   size_t numTasks  = 128)
                : Scheduler("System", numWorker, numTasks)
                , Config(config) {
                Logger::clear();
            }
        };

    };  // namespace system
};      // namespace corekit
