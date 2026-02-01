#pragma once
#include <type_traits>

namespace corekit {

    namespace system {

        template <typename... Params>
        struct Config : public Params... {
            template <typename... Others>
            static Config<Others...> merge(const Others&... others) {
                static_assert(
                    (std::is_base_of_v<Config<>, Others> && ...),
                    "Config::merge => all others must derive from Config<>.");

                return Config<Others...>{static_cast<const Others&>(others)...};
            }
        };

        using BaseConfig = Config<>;

    };  // namespace system
};  // namespace corekit
