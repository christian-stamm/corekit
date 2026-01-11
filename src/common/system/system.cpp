#include "corekit/system/system.hpp"

#include <nlohmann/json_fwd.hpp>

#include "corekit/system/diagnostics/assert.hpp"

namespace corekit {

    namespace system {

        Manager::Manager(const Settings& settings)
            : worker(settings.numWorker)
            , workdir(settings.workdir)
            , logger("System") {
            std::system("clear");
        }

        Manager& Manager::operator<<(const Settings& config) {
            std::destroy_at(&worker);
            std::construct_at(&worker, config.numWorker);

            workdir = config.workdir;
            return *this;
        }

        void Manager::shutdown() {
            killreq.request_stop();
        }

        bool Manager::ok() const {
            return !killreq.stop_requested();
        }

        Hash getEnv(const Hash& key) {
            const char* value = std::getenv(key.c_str());
            corecheck(value != nullptr, "Environment variable not set: " + key);
            return value;
        }

    }  // namespace system

}  // namespace corekit
