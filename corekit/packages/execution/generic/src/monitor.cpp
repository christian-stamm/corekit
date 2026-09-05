#include "corekit/monitor.hpp"

#include "corekit/time.hpp"

namespace corekit {

    Monitor::Monitor() : Task("Monitor") {}

    VoidResult Monitor::on_run(StopToken token) {
        while (!token.stop_requested()) {
            while (!Error::stack.empty()) {
                const Error* error = Error::stack.top();
                logger.error() << "\n" << error->what() << "\n";
                Error::stack.pop();
            }

            Time::sleep(1);
            logger() << "Heartbeat...";
        }

        return VoidResult();
    }

}  // namespace corekit