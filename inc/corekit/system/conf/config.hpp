#pragma once
#include <tuple>
#include <type_traits>
#include <utility>

namespace corekit {
    namespace system {

        template <typename... Params>
        struct Config {
            using Settings = Config<Params...>;

           public:
            Config() : params() {}
            Config(Params... params) noexcept  //
                : params(std::forward<Params>(params)...) {}

            Config(const Settings& other) : params(other.params) {}

            template <typename T>
            T& get() {
                static_assert(contains_all<T>(), "Type not in Config");
                return std::get<T>(this->params);
            }

            template <typename T>
            const T& get() const {
                static_assert(contains_all<T>(), "Type not in Config");
                return std::get<T>(this->params);
            }

            template <typename... T>
            Config<T...> fetch() const {
                static_assert(contains_all<T...>(), "Type not in Config");
                return Config<T...>(std::get<T>(this->params)...);
            }

           protected:
            template <typename... T>
            static constexpr bool contains_all() {
                return (contains_any<T>() && ...);
            }

            template <typename T>
            static constexpr bool contains_any() {
                return (std::is_same_v<T, Params> || ...);
            }

            std::tuple<Params...> params;
        };

    };  // namespace system
};  // namespace corekit