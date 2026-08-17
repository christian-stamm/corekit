#pragma once
#include <memory>
#include <type_traits>

#include "corekit/system/config.hpp"
#include "corekit/system/runtime.hpp"

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

            using RuntimePtr = std::shared_ptr<State>;

            Context()                     = delete;
            Context(const Context &other) = default;

            template <typename... Params>
            Context(RuntimePtr rt, Params... params)
                : Context(rt, Settings(params...)) {}

            Context(RuntimePtr rt, const Settings &cfg)
                : Settings(cfg)
                , rt(rt) {
                corecheck(rt != nullptr,
                          "Context => Runtime pointer cannot be null.");
            }

            template <typename TargetContext, typename... ExtraParams>
            const TargetContext as(ExtraParams... params) const {
                static_assert(xtd::is_context_v<TargetContext>,
                              "Context => as() requires a Context type.");

                return TargetContext(rt, Settings::merge(Config(params...)));
            }

            template <typename TargetContext, typename... ExtraParams>
            const TargetContext as(const Config<ExtraParams...> &addon =
                                       Config<ExtraParams...>()) const {
                static_assert(xtd::is_context_v<TargetContext>,
                              "Context => as() requires a Context type.");

                return TargetContext(rt, Settings::merge(addon));
            }

            const RuntimePtr rt;
        };

    };  // namespace system
};  // namespace corekit