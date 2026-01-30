#pragma once
#include <type_traits>

#include "corekit/system/flow/scheduler.hpp"
#include "corekit/types.hpp"
#include "corekit/utils/device.hpp"
#include "corekit/utils/watch.hpp"

namespace corekit {

    namespace system {

        using namespace corekit::types;
        using namespace corekit::utils;

        Hash getEnv(const Hash& key);

        struct BaseConfig {
            size_t numWorker = 4;
            size_t numTasks  = 128;
        };

        template <typename Config = BaseConfig>
        class System
            : public Scheduler
            , public Config {
           public:
            static_assert(std::is_base_of<BaseConfig, Config>::value,
                          "Config must derive from BaseConfig");

            System(Config config = {})
                : Scheduler("System", config.numWorker, config.numTasks)
                , Config(config) {
                // Logger::clear();
            }
        };

    };  // namespace system
};      // namespace corekit
