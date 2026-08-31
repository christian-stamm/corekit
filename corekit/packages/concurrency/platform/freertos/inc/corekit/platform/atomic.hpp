#pragma once

#include <FreeRTOS.h>
#include <semphr.h>
#include <task.h>

#include <atomic>
#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace corekit::platform {

    template <typename T>
    class Atomic {
        public:

            using Ptr       = std::shared_ptr<Atomic<T>>;
            using ValueType = T;

            explicit Atomic(T value = T())
                : value_(value) { }

            ~Atomic() { }

            Atomic(const Atomic&)            = delete;
            Atomic(Atomic&&)                 = delete;
            Atomic& operator=(const Atomic&) = delete;
            Atomic& operator=(Atomic&&)      = delete;

            T load(std::memory_order = std::memory_order_seq_cst) const noexcept {
                return with_lock([this] { return value_; });
            }

            void store(T desired, std::memory_order = std::memory_order_seq_cst) noexcept {
                with_lock([this, desired] { value_ = desired; });
            }

            T exchange(T desired, std::memory_order = std::memory_order_seq_cst) noexcept {
                return with_lock([this, desired] {
                    T old  = value_;
                    value_ = desired;
                    return old;
                });
            }

            bool compare_exchange(T& expected, T desired, std::memory_order = std::memory_order_seq_cst, std::memory_order = std::memory_order_seq_cst) noexcept {
                return with_lock([this, &expected, desired] {
                    if (value_ == expected) {
                        value_ = desired;
                        return true;
                    }
                    expected = value_;
                    return false;
                });
            }

            template <typename U = T>
            requires((std::integral<U> && !std::same_as<U, bool>) || std::floating_point<U>)
            U fetch_add(U arg, std::memory_order = std::memory_order_seq_cst) noexcept {
                return with_lock([this, arg] {
                    U old   = value_;
                    value_ += arg;
                    return old;
                });
            }

            template <typename U = T>
            requires std::is_pointer_v<U>
            U fetch_add(std::ptrdiff_t arg, std::memory_order = std::memory_order_seq_cst) noexcept {
                return with_lock([this, arg] {
                    U old   = value_;
                    value_ += arg;
                    return old;
                });
            }

            template <typename U = T>
            requires((std::integral<U> && !std::same_as<U, bool>) || std::floating_point<U>)
            U fetch_sub(U arg, std::memory_order = std::memory_order_seq_cst) noexcept {
                return with_lock([this, arg] {
                    U old   = value_;
                    value_ -= arg;
                    return old;
                });
            }

            template <typename U = T>
            requires std::is_pointer_v<U>
            U fetch_sub(std::ptrdiff_t arg, std::memory_order = std::memory_order_seq_cst) noexcept {
                return with_lock([this, arg] {
                    U old   = value_;
                    value_ -= arg;
                    return old;
                });
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U fetch_and(U arg, std::memory_order = std::memory_order_seq_cst) noexcept {
                return with_lock([this, arg] {
                    U old   = value_;
                    value_ &= arg;
                    return old;
                });
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U fetch_or(U arg, std::memory_order = std::memory_order_seq_cst) noexcept {
                return with_lock([this, arg] {
                    U old   = value_;
                    value_ |= arg;
                    return old;
                });
            }

            template <typename U = T>
            requires(std::integral<U> && !std::same_as<U, bool>)
            U fetch_xor(U arg, std::memory_order = std::memory_order_seq_cst) noexcept {
                return with_lock([this, arg] {
                    U old   = value_;
                    value_ ^= arg;
                    return old;
                });
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

            template <typename F>
            decltype(auto) with_lock(F&& fn) const noexcept {
                if (xPortIsInsideInterrupt()) {
                    const UBaseType_t state = taskENTER_CRITICAL_FROM_ISR();
                    if constexpr (std::is_void_v<std::invoke_result_t<F>>) {
                        fn();
                        taskEXIT_CRITICAL_FROM_ISR(state);
                        return;
                    } else {
                        auto result = fn();
                        taskEXIT_CRITICAL_FROM_ISR(state);
                        return result;
                    }
                }

                taskENTER_CRITICAL();
                if constexpr (std::is_void_v<std::invoke_result_t<F>>) {
                    fn();
                    taskEXIT_CRITICAL();
                    return;
                } else {
                    auto result = fn();
                    taskEXIT_CRITICAL();
                    return result;
                }
            }

            mutable T value_{};
    };

    extern template class Atomic<bool>;
    extern template class Atomic<unsigned int>;
    extern template class Atomic<int>;

} // namespace corekit::platform
