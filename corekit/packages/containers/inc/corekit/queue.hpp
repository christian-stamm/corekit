#pragma once
#include <mutex>
#include <queue>

#include "corekit/conditionvariable.hpp"
#include "corekit/mutex.hpp"
#include "corekit/result.hpp"

namespace corekit {

    constexpr size_t MAX_QUEUE_WAITER = 10;

    template <typename T>
    class Queue {
       public:
        explicit Queue(size_t capacity)
            : capacity_(capacity)
            , producer_(MAX_QUEUE_WAITER)
            , consumer_(MAX_QUEUE_WAITER) {}

        VoidResult push(T item, bool wait = true) {
            {
                std::unique_lock lock(mutex_);

                if (wait) {
                    producer_.wait(lock, [this] { return !unsafe_full(); });
                } else if (unsafe_full()) {
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
                    consumer_.wait(lock, [this] { return !unsafe_empty(); });
                } else if (unsafe_empty()) {
                    return RuntimeError("Queue is empty");
                }

                item = std::move(queue_.front());
                queue_.pop();
            }

            producer_.notify_one();
            return item;
        }

        void clear() {
            std::lock_guard lock(mutex_);
            std::queue<T>   empty;
            std::swap(queue_, empty);
        }

        bool empty() const {
            std::lock_guard lock(mutex_);
            return queue_.empty();
        }

        size_t size() const {
            std::lock_guard lock(mutex_);
            return queue_.size();
        }

        bool full() const {
            std::lock_guard lock(mutex_);
            return unsafe_full();
        }

       private:
        inline bool unsafe_full() const {
            return capacity_ <= queue_.size();
        }

        inline bool unsafe_empty() const {
            return queue_.empty();
        }

        size_t        capacity_;
        std::queue<T> queue_;

        mutable Mutex     mutex_;
        ConditionVariable producer_;
        ConditionVariable consumer_;
    };

    extern template class Queue<int>;
    extern template class Queue<uint>;

}  // namespace corekit