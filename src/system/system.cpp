#include <nlohmann/json_fwd.hpp>

#include "corekit/core.hpp"

namespace corekit {

    namespace system {

        Hash getEnv(const Hash& key) {
            const char* value = std::getenv(key.c_str());
            corecheck(value != nullptr, "Environment variable not set: " + key);
            return value;
        }

    }  // namespace system

}  // namespace corekit
