#pragma once

#include "corekit/render/program.hpp"
#include "corekit/render/shader.hpp"
#include "corekit/render/texture.hpp"
#include "corekit/render/window.hpp"
#include "corekit/system/conf/config.hpp"
#include "corekit/system/conf/context.hpp"
#include "corekit/system/conf/observer.hpp"
#include "corekit/types.hpp"
#include "corekit/utils/assert.hpp"
#include "corekit/utils/device.hpp"
#include "corekit/utils/filemgr.hpp"
#include "corekit/utils/logger.hpp"
#include "corekit/utils/math.hpp"
#include "corekit/utils/watch.hpp"

namespace corekit {

    namespace utils {
        using namespace corekit::types;
    };  // namespace utils

    namespace render {
        using namespace corekit::utils;
    };  // namespace render

    namespace system {
        using namespace corekit::utils;

        static Hash getEnv(const Name& key) {
            const char* value = std::getenv(key.c_str());
            corecheck(value != nullptr, "Environment variable not set: " + key);
            return value;
        }

    };  // namespace system

};  // namespace corekit
