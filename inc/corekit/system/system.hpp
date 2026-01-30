#pragma once
#include "corekit/system/context.hpp"
#include "corekit/system/flow/scheduler.hpp"
#include "corekit/types.hpp"

namespace corekit {

    namespace system {

        using namespace corekit::types;
        using namespace corekit::utils;

        class System {
           public:
            struct Settings {
                size_t numWorkers = 4;
                size_t numTasks   = 64;
            };

            System(Settings config)
                : config(config)
                , scheduler(killreq, config.numWorkers, config.numTasks)
                , context(this->config, this->scheduler, this->killreq) {
                Logger::clear();
            }

            const Context<Settings>& getContext() const {
                return context;
            }

            static Hash getEnv(const Name& key);

           private:
            Settings          config;
            Killreq           killreq;
            Scheduler         scheduler;
            Context<Settings> context;
        };

    };  // namespace system
};  // namespace corekit

namespace nlohmann {
    using namespace corekit::types;
    using namespace corekit::system;

    static void to_json(JsonMap& j, const System::Settings& cfg) {
        j = JsonMap{
            {"numWorkers", cfg.numWorkers},
            {"numTasks", cfg.numTasks},
        };
    }

    static void from_json(const JsonMap& j, System::Settings& cfg) {
        j.at("numWorkers").get_to(cfg.numWorkers);
        j.at("numTasks").get_to(cfg.numTasks);
    }

};  // namespace nlohmann
