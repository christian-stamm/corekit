#pragma once

#include "corekit/system/flow/scheduler.hpp"
#include "corekit/types.hpp"

namespace corekit {
    namespace system {

        template <typename Config>
        struct Context {
            Context(Config& config, Scheduler& scheduler, Killreq& killreq)
                : cfg(config)
                , mgr(scheduler)
                , sig(killreq) {}

            template <typename Module>
            Context<Module> cast(Module& config) const {
                return Context<Module>(config, mgr, sig);
            }

            bool ok() const {
                return !sig.stop_requested();
            }

            void kill() const {
                if (!sig.stop_requested()) {
                    sig.request_stop();
                }
            }

            Config&    cfg;
            Scheduler& mgr;

           protected:
            Killreq& sig;
        };

    };  // namespace system
};  // namespace corekit