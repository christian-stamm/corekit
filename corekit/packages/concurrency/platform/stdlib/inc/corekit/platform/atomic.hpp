#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace corekit::platform {

    using uint = unsigned int;

    template <typename T>
    class Atomic {
        public:

            using Ptr                   = std::shared_ptr<Atomic<T>>;
            using ValueType             = T;

            constexpr Atomic() noexcept = default;

            constexpr explicit Atomic(T desired) noexcept
                : value_(desired) { }

            Atomic(const Atomic&)            = delete;
            Atomic(Atomic&&)                 = delete;
            Atomic& operator=(const Atomic&) = delete;
            Atomic& operator=(Atomic&&)      = delete;

            T load(std::memory_order order = std::memory_order_seq_cst) const noexcept { return value_.load(order); }

            void store(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept { value_.store(desired, order); }

            T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept { return value_.exchange(desired, order); }

            bool compare_exchange(T& expected, T desired, std::memory_order success = std::memory_order_seq_cst, std::memory_order failure = std::memory_order_seq_cst) noexcept {
                return value_.compare_exchange_strong(expected, desired, success, failure);
            }

            void wait(T old, std::memory_order order = std::memory_order_seq_cst) const noexcept { value_.wait(old, order); }

            void notify_one() noexcept { value_.notify_one(); }

            void notify_all() noexcept { value_.notify_all(); }

            template <typename U = T>
            requires((std::integral<U> && !std::same_as<U, bool>) || std::floating_point<U>)
            U fetch_add(U arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
                return value_.fetch_add(arg, order);
            }

            template <typename U = T>
            requires std::is_pointer_v<U>
            U fetch_add(std::ptrdiff_t arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
                return value_.fetch_add(arg, order);
            }

            template <typename U = T>
            requires((std::integral<U> && !std::same_as<U, bool>) || std::floating_point<U>)
            U fetch_sub(U arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
                return value_.fetch_sub(arg, order);
            }

            template <typename U = T>
            requires std::is_pointer_v<U>
            U fetch_sub(std::ptrdiff_t arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
                return value_.fetch_sub(arg, order);
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U fetch_and(U arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
                return value_.fetch_and(arg, order);
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U fetch_or(U arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
                return value_.fetch_or(arg, order);
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U fetch_xor(U arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
                return value_.fetch_xor(arg, order);
            }

            template <typename U = T>
            requires((std::integral<U> && !std::same_as<U, bool>) || std::floating_point<U>)
            U operator+=(U arg) noexcept {
                return fetch_add(arg) + arg;
            }

            template <typename U = T>
            requires std::is_pointer_v<U>
            U operator+=(std::ptrdiff_t arg) noexcept {
                return fetch_add(arg) + arg;
            }

            template <typename U = T>
            requires((std::integral<U> && !std::same_as<U, bool>) || std::floating_point<U>)
            U operator-=(U arg) noexcept {
                return fetch_sub(arg) - arg;
            }

            template <typename U = T>
            requires std::is_pointer_v<U>
            U operator-=(std::ptrdiff_t arg) noexcept {
                return fetch_sub(arg) - arg;
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U operator&=(U arg) noexcept {
                return fetch_and(arg) & arg;
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U operator|=(U arg) noexcept {
                return fetch_or(arg) | arg;
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U operator^=(U arg) noexcept {
                return fetch_xor(arg) ^ arg;
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U operator++() noexcept {
                return fetch_add(U{1}) + U{1};
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U operator++(int) noexcept {
                return fetch_add(U{1});
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U operator--() noexcept {
                return fetch_sub(U{1}) - U{1};
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U operator--(int) noexcept {
                return fetch_sub(U{1});
            }

        private:

            std::atomic<T> value_{};
    };

    extern template class Atomic<bool>;
    extern template class Atomic<uint>;
    extern template class Atomic<int>;

} // namespace corekit::platform
