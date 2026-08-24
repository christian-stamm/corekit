#pragma once
#include <concepts>

namespace corekit {

    template <typename T>
    concept TimeType = requires(T t) {
        { T::sleep(1.0f) } -> std::same_as<void>;
        { T::now() } -> std::same_as<double>;
    };

    template <typename T, typename VType>
    concept AtomicType =
        requires(T a, VType value, VType expected, VType desired) {
            { a.load() } -> std::same_as<VType>;
            { a.store(value) } -> std::same_as<void>;
            {
                a.compare_exchange_strong(expected, desired)
            } -> std::same_as<bool>;
        };

    template <typename T>
    concept MutexType = requires(T m) {
        { m.lock() } -> std::same_as<void>;
        { m.unlock() } -> std::same_as<void>;
        { m.try_lock() } -> std::same_as<bool>;
    };

    template <typename T>
    concept SemaphoreType = requires(T s) {
        { s.acquire() } -> std::same_as<void>;
        { s.release() } -> std::same_as<void>;
        { s.try_acquire() } -> std::same_as<bool>;
    };

    template <class T>
    concept StopTokenType = requires(const T& ctoken) {
        { ctoken.stop_requested() } -> std::same_as<bool>;
        { ctoken.stop_possible() } -> std::same_as<bool>;
    };

    template <class T, typename Token>
    concept StopSourceType =
        StopTokenType<Token> && requires(T& source, const T& csource) {
            { source.request_stop() } -> std::same_as<bool>;
            { csource.stop_requested() } -> std::same_as<bool>;
            { csource.stop_possible() } -> std::same_as<bool>;
            { csource.get_token() } -> std::same_as<Token>;
        };

};  // namespace corekit