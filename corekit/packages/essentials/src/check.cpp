#include "corekit/check.hpp"

#include "corekit/logger.hpp"
#include "corekit/mutex.hpp"

namespace corekit {

    const Logger syslog("CORE");
    const Mutex  syslock;

    bool corecheck(bool condition, const Error& error) {
        // Perform a core check to ensure that the system is in a valid state.
#ifndef NDEBUG
        if (!condition) {
            syslog() << error.what();
            return false;
        }
#endif
        return true;
    }

    bool corecheck() {
        // Perform a core check to ensure that the system is in a valid state.
#ifndef NDEBUG

        if (!Error::stack.empty()) {
            std::lock_guard lock(syslock);

            while (!Error::stack.empty()) {
                const Error* error = Error::stack.top();

                if (error) {
                    syslog() << "Core check failed: " << error->what();
                }

                Error::stack.pop();
            }

            return false;
        }

#endif

        return true;
    }

}  // namespace corekit