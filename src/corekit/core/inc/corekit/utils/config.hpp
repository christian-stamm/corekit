#pragma once
#include <type_traits>

namespace corekit {
    namespace utils {

        template <typename... Params>
        struct Config : public Params... {
            using Tag = void;  // Tag to identify Config types

           public:
            Config() = default;

            Config(Params... params) noexcept requires(0 < (sizeof...(Params)))
                : Params(params)... {}

            template <typename... Args>
            Config(const Config<Args...>& other) noexcept
                requires((std::is_base_of_v<Params, Config<Args...>> && ...))
                : Params(static_cast<const Params&>(other))... {}

            template <typename... Args>
            static const Config<Args...> build(Args... args) {
                return Config<Args...>(args...);
            }

            template <typename... Args>
            const Config<Args...> select() const {
                return Config<Args...>(*this);
            }

            template <typename... Args>
            const Config<Params..., Args...> extend(Args... args) {
                return Config<Params..., Args...>(static_cast<Params>(*this)...,
                                                  static_cast<Args>(args)...);
            }

            template <typename... Args>
            const Config<Params..., Args...> merge(
                const Config<Args...>& other) const {
                return Config<Params..., Args...>(static_cast<Params>(*this)...,
                                                  static_cast<Args>(other)...);
            };
        };

        namespace xtd {

            template <typename T, typename = void>
            struct is_config : std::false_type {};

            template <typename T>
            struct is_config<T, std::void_t<typename T::Tag>>
                : std::true_type {};

            template <typename T>
            inline constexpr bool is_config_v = is_config<T>::value;

        };  // namespace xtd

    };  // namespace utils
};      // namespace corekit