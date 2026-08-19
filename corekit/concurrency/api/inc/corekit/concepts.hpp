#pragma once
#include <concepts>

namespace corekit {

    template <typename T>
    concept AtomicType = requires(T                     a,
                                  typename T::ValueType value,
                                  typename T::ValueType expected,
                                  typename T::ValueType desired) {
        { a.load() } -> std::same_as<typename T::ValueType>;
        { a.store(value) } -> std::same_as<void>;
        { a.compare_exchange_strong(expected, desired) } -> std::same_as<bool>;
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

    template <typename T, typename Token>
    concept TaskType = requires(T t, const Token& token) {
        { t.on_enter(token) } -> std::same_as<bool>;
        { t.mainloop(token) } -> std::same_as<bool>;
        { t.on_leave(token) } -> std::same_as<bool>;
    };

    template <typename T, typename Token>
    concept ThreadType = StopTokenType<Token> && requires(T t, Token token) {
        { t.run() } -> std::same_as<void>;
        { t.join() } -> std::same_as<void>;
        { t.request_stop() } -> std::same_as<void>;
    };

};  // namespace corekit