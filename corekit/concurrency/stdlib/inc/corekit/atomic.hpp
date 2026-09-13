#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace corekit {

    using uint = unsigned int;

    template <typename T>
    class Atomic : public std::atomic<T> {
        public:

            using Ptr = std::shared_ptr<Atomic<T>>;

            constexpr Atomic() noexcept = default;

            constexpr explicit Atomic(T desired) noexcept
                : std::atomic<T>(desired) { }

            Atomic(const Atomic&)            = delete;
            Atomic(Atomic&&)                 = delete;
            Atomic& operator=(const Atomic&) = delete;
            Atomic& operator=(Atomic&&)      = delete;
    };

    extern template class Atomic<bool>;
    extern template class Atomic<uint>;
    extern template class Atomic<int>;

} // namespace corekit
