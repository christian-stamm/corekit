#pragma once
#include <type_traits>

#include "corekit/system/conf/config.hpp"
#include "corekit/system/conf/runtime.hpp"

namespace corekit {

    namespace system {

        using namespace corekit::types;
        using namespace corekit::utils;

        template <typename State = Runtime, typename... Params>
        struct Context {
            static_assert(std::is_base_of_v<Runtime, State>,
                          "Context => RuntimeParams must derive from Runtime.");

            using RuntimeParams = State;
            using StaticParams  = const Config<Params...>;

            template <typename... SubParams>
            Context<RuntimeParams, SubParams...> pick() {
                return Context<RuntimeParams, SubParams...>{
                    rt,
                    cfg.template fetch<SubParams...>(),
                };
            }

            RuntimeParams& rt;
            StaticParams   cfg;
        };

    };  // namespace system
};  // namespace corekit
