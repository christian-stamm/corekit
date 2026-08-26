#pragma once
#include <concepts>

namespace corekit {

    template <typename T, typename... Configs>
    inline constexpr std::size_t count_type_v =
        (std::size_t{0} + ... + std::size_t{std::same_as<T, Configs>});

    template <typename... Configs>
    class Config : public Configs... {
        static_assert(((count_type_v<Configs, Configs...> == 1) && ...),
                      "Each config type may only appear once");

       public:
        template <typename T>
            requires((std::same_as<T, Configs> || ...))
        constexpr T& get() noexcept {
            return static_cast<T&>(*this);
        }

        template <typename T>
            requires((std::same_as<T, Configs> || ...))
        constexpr const T& get() const noexcept {
            return static_cast<const T&>(*this);
        }
    };

}  // namespace corekit