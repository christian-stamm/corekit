#pragma once

#include "corekit/platform/stack.hpp"

namespace corekit {

    template <typename T>
    using Stack = platform::Stack<T>;

    template <typename Stack, typename Item>
    concept StackLike = requires(Stack s, Item i) {
        { s.push(i, false) } -> std::convertible_to<bool>;
        { s.pop(i, false) } -> std::convertible_to<bool>;
    };

}  // namespace corekit