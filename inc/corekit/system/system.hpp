#pragma once
#include <type_traits>

#include "corekit/system/context.hpp"
#include "corekit/system/flow/scheduler.hpp"
#include "corekit/types.hpp"

namespace corekit {

    namespace system {

        using namespace corekit::types;
        using namespace corekit::utils;

        struct SysCfg {
            Scheduler::Settings shedulerCfg;

            static Hash getEnv(const Name& key);
        };

        template <typename Config = SysCfg>
        class System {
           public:
            static_assert(std::is_base_of<SysCfg, Config>::value,
                          "Config must derive from System::Settings");

            System(Config& config)
                : scheduler(config.shedulerCfg, killreq)
                , context(config, scheduler, killreq) {
                Logger::clear();
            }

            Context<Config>& ctx() {
                return context;
            }

           private:
            Killreq         killreq;
            Scheduler       scheduler;
            Context<Config> context;
        };

    };  // namespace system
};  // namespace corekit
