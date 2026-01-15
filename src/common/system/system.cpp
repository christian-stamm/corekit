#include "corekit/system/system.hpp"

#include <nlohmann/json_fwd.hpp>

#include "corekit/system/diagnostics/assert.hpp"

namespace corekit {

    namespace system {

        Manager::Manager(const Settings& settings)
            : Device("System")
            , scheduler(std::make_shared<Scheduler>(settings.numWorker))
            , workdir(settings.workdir) {
            Logger::clear();
        }

        Manager& Manager::operator<<(const Settings& config) {
            if (scheduler) {
                scheduler->kill();
                scheduler.reset();
            }

            scheduler = std::make_shared<Scheduler>(config.numWorker);
            workdir   = config.workdir;

            return *this;
        }

        void Manager::shutdown() {
            killreq.request_stop();
        }

        bool Manager::ok() const {
            return !killreq.stop_requested();
        }

        float Manager::time() const {
            return clock.elapsed();
        }

        Hash getEnv(const Hash& key) {
            const char* value = std::getenv(key.c_str());
            corecheck(value != nullptr, "Environment variable not set: " + key);
            return value;
        }

    }  // namespace system

}  // namespace corekit
