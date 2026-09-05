#include "corekit/check.hpp"

#include "corekit/logger.hpp"
#include "corekit/mutex.hpp"

namespace corekit {

    const Logger syslog("CORE");

    bool corecheck(bool condition, const Error& error) {
        // Perform a core check to ensure that the system is in a valid state.
#ifndef NDEBUG
        if (!condition) {
            syslog.error() << error.what();
            return false;
        }
#endif
        return true;
    }

    bool corecheck() {
        // Perform a core check to ensure that the system is in a valid state.
        // #ifndef NDEBUG

        if (!Error::stack.empty()) {
            LogStream log = syslog.error();
            Error::stack.dump(log);
            return false;
        }

        // #endif

        return true;
    }

}  // namespace corekit