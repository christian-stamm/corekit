#pragma once
#include <mutex>
#include <queue>

#include "corekit/assert.hpp"
#include "corekit/conditionvariable.hpp"
#include "corekit/mutex.hpp"

namespace corekit {

    template <typename T>
    class Queue {
       public:
        explicit Queue(size_t capacity) : capacity_(capacity) {}

        VoidResult push(T item, bool wait = true) {
            {
                std::unique_lock lock(mutex_);

                if (wait) {
                    producer_.wait(lock, [this] { return !is_full(); });
                } else if (is_full()) {
                    return RuntimeError("Queue is full");
                }

                queue_.push(std::move(item));
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
                    return RuntimeError("Queue is empty");
                }

                item = std::move(queue_.front());
                queue_.pop();
            }

            producer_.notify_one();
            return item;
        }

       private:
        inline bool is_full() const {
            return capacity_ <= queue_.size();
        }

        inline bool is_empty() const {
            return queue_.empty();
        }

        size_t        capacity_;
        std::queue<T> queue_;

        Mutex             mutex_;
        ConditionVariable producer_;
        ConditionVariable consumer_;
    };

    extern template class Queue<int>;
    extern template class Queue<uint>;

}  // namespace corekit