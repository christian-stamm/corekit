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
        { a.compare_exchange(expected, desired) } -> std::same_as<bool>;
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

    template <typename T>
    concept ThreadType = requires(T t) {
        { t.run() } -> std::same_as<void>;
        { t.detach() } -> std::same_as<void>;
        { t.join() } -> std::same_as<void>;
        { t.joinable() } -> std::same_as<bool>;
    };

};  // namespace corekit