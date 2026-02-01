#pragma once

#include <type_traits>

#include "corekit/system/config.hpp"
#include "corekit/system/flow/scheduler.hpp"
#include "corekit/types.hpp"

namespace corekit {
    namespace system {

        template <typename Params = BaseConfig>
        struct Context {
            static_assert(std::is_base_of<BaseConfig, Params>::value,
                          "Params must derive from BaseConfig.");

            Context(Params& config, Scheduler& scheduler, Killreq& killreq)
                : cfg(config)
                , mgr(scheduler)
                , signal(killreq) {}

            template <typename OtherParams>
            Context<OtherParams> as(OtherParams& params) const {
                return Context<OtherParams>(params, mgr, signal);
            }

            bool ok() const {
                return !signal.stop_requested();
            }

            void kill() const {
                if (!signal.stop_requested()) {
                    signal.request_stop();
                }
            }

            Params&    cfg;
            Scheduler& mgr;

           protected:
            Killreq& signal;
        };

    };  // namespace system
};  // namespace corekit