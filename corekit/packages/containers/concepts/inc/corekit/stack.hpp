#pragma once

#include "corekit/platform/stack.hpp"

namespace corekit {

    template <typename T>
    using Stack = platform::Stack<T>;

    template <typename Stack, typename Item>
    concept StackLike = requires(Stack s, Item i) {
        { s.push(i, false) } -> std::convertible_to<VoidResult>;
        { s.pop(i, false) } -> std::convertible_to<Result<Item>>;
    };

}  // namespace corekit