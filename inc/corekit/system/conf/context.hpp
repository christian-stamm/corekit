#pragma once
#include <memory>
#include <type_traits>

#include "corekit/system/conf/config.hpp"
#include "corekit/system/conf/runtime.hpp"

namespace corekit {
    namespace system {

        namespace xtd {

            template <typename T, typename = void>
            struct is_context : std::false_type {};

            template <typename T>
            struct is_context<T, std::void_t<typename T::Tag>>
                : std::true_type {};

            template <typename T>
            inline constexpr bool is_context_v = is_context<T>::value;

        };  // namespace xtd

        template <typename State = Runtime, typename Settings = Config<>>
        struct Context : public Settings {
            static_assert(std::is_base_of_v<corekit::system::Runtime, State>,
                          "Context => State must be derived from Runtime.");

            static_assert(xtd::is_config_v<Settings>,
                          "Context => Settings must be derived from Config<>.");

            template <typename otherstate, typename otherconfig>
            friend struct Context;

            using Runtime = std::shared_ptr<State>;

            Context() = delete;

            template <typename... Params>
            static const Context build(Runtime rt, Params... params) {
                return Context(params..., rt);
            }

            static const Context build(Runtime rt, const Settings &cfg) {
                return Context(cfg, rt);
            }

            template <typename TargetContext = Context>
            const TargetContext as() const {
                static_assert(xtd::is_context_v<TargetContext>,
                              "Context => as() requires a Context type.");

                return TargetContext(*this, rt);
            }

            template <typename TargetContext = Context, typename... ExtraParams>
            const TargetContext as(ExtraParams... others) const {
                static_assert(xtd::is_context_v<TargetContext>,
                              "Context => as() requires a Context type.");

                return TargetContext(Settings::merge(Config(others...)), rt);
            }

            const Runtime rt;

           protected:
            Context(const Settings &cfg, Runtime rt) : Settings(cfg), rt(rt) {}
        };

    };  // namespace system
};  // namespace corekit