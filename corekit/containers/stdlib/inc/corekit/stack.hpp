#pragma once

#include "corekit/queue.hpp"

namespace corekit {

    template <typename T>
    class Stack : public Queue<T> {
        public:

            explicit Stack(size_t capacity)
                : Queue<T>(capacity) { }

            virtual bool try_pop(T& item, Timeout timeout = std::nullopt) {
                std::unique_lock lock(mutex_);

                if (timeout.has_value()) {
                    if (!consumer_.wait_for(lock, std::chrono::duration<double>(timeout.value()), [this] {
                            return !unsafe_empty();
                        })) {
                        return false; // timeout
                    }
                } else {
                    consumer_.wait(lock, [this] { return !unsafe_empty(); });
                }

                item = std::move(queue_.back());
                queue_.pop_back();
                producer_.notify_one();
                return true;
            }
    };

    extern template class Stack<int>;
    extern template class Stack<uint>;

} // namespace corekit