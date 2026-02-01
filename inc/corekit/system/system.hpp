#pragma once

#include <format>

#include "corekit/system/config.hpp"
#include "corekit/system/context.hpp"
#include "corekit/system/flow/scheduler.hpp"
#include "corekit/types.hpp"
#include "corekit/utils/device.hpp"
#include "corekit/utils/logger.hpp"

namespace corekit {

    namespace system {

        using namespace corekit::types;
        using namespace corekit::utils;

        static Hash getEnv(const Name& key);

        struct SysConf : public BaseConfig {
            Scheduler::Settings shedulerCfg;
        };

        template <typename Config = SysConf>
        class System : public Device {
           public:
            static_assert(std::is_base_of_v<SysConf, Config>,
                          "Config must derive from SysConf.");

            System(Config& config)
                : Device("System")
                , scheduler(config.shedulerCfg, killreq)
                , ctx(config, scheduler, killreq) {}

            void shutdown() {
                ctx.kill();
            }

            const Context<Config> ctx;

           protected:
            virtual bool prepare() override {
                Logger::clear();
                logger() << std::format("Launched with {} workers",
                                        scheduler.numWorkers());

                return ctx.ok();
            }

            virtual bool cleanup() override {
                logger() << "Shutdown...";
                shutdown();
                return true;
            }

           private:
            Killreq   killreq;
            Scheduler scheduler;
        };

    };  // namespace system
};  // namespace corekit
