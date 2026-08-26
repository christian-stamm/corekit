#pragma once
#include <mutex>
#include <stack>

#include "corekit/assert.hpp"
#include "corekit/conditionvariable.hpp"
#include "corekit/mutex.hpp"

namespace corekit {

    template <typename T>
    class Stack {
       public:
        explicit Stack(size_t capacity) : capacity_(capacity) {}

        VoidResult push(T item, bool wait = true) {
            {
                std::unique_lock lock(mutex_);

                if (wait) {
                    producer_.wait(lock, [this] { return !is_full(); });
                } else if (is_full()) {
                    return RuntimeError("Stack is full");
                }

                stack_.push(std::move(item));
            }

            consumer_.notify_one();
            return VoidResult();
        }

        Result<T> pop(T& item, bool wait = true) {
            {
                std::unique_lock lock(mutex_);

                if (wait) {
                    consumer_.wait(lock, [this] { return !is_empty(); });
                } else if (is_empty()) {
                    return RuntimeError("Stack is empty");
                }

                item = std::move(stack_.top());
                stack_.pop();
            }

            producer_.notify_one();
            return item;
        }

       private:
        inline bool is_full() const {
            return capacity_ <= stack_.size();
        }

        inline bool is_empty() const {
            return stack_.empty();
        }

        size_t        capacity_;
        std::stack<T> stack_;

        Mutex             mutex_;
        ConditionVariable producer_;
        ConditionVariable consumer_;
    };

    extern template class Stack<int>;
    extern template class Stack<uint>;

}  // namespace corekit