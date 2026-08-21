#pragma once

#include <concepts>

namespace corekit {

    template <typename T, typename V>
    concept AtomicLike =  //
        requires(T a, V value, V expected, V desired) {
            { a.load() } -> std::convertible_to<V>;
            { a.store(value) } -> std::convertible_to<void>;
            {
                a.compare_exchange_strong(expected, desired)
            } -> std::convertible_to<bool>;
        };

    template <typename T>
    concept Lockable =  //
        requires(T m) {
            { m.lock() } -> std::convertible_to<void>;
            { m.unlock() } -> std::convertible_to<void>;
            { m.try_lock() } -> std::convertible_to<bool>;
        };

    template <typename T>
    concept SemaphoreLike =  //
        requires(T s) {
            { s.acquire() } -> std::convertible_to<void>;
            { s.release() } -> std::convertible_to<void>;
            { s.try_acquire() } -> std::convertible_to<bool>;
        };

    template <typename T>
    concept StopTokenLike =  //
        requires(const T& token) {
            { token.stop_requested() } -> std::convertible_to<bool>;
            { token.stop_possible() } -> std::convertible_to<bool>;
        };

    template <typename T, typename Token>
    concept StopSourceLike =  //
        StopTokenLike<Token> && requires(T& source, const T& csource) {
            { source.request_stop() } -> std::convertible_to<bool>;
            { csource.stop_requested() } -> std::convertible_to<bool>;
            { csource.stop_possible() } -> std::convertible_to<bool>;
            { csource.get_token() } -> std::convertible_to<Token>;
        };

}  // namespace corekit