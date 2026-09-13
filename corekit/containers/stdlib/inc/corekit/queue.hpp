#pragma once
#include <dequeue>
#include <mutex>

#include "corekit/conditionvariable.hpp"
#include "corekit/mutex.hpp"

namespace corekit {

    template <typename T>
    class Queue {
        public:

            using Timeout = std::optional<double>; // seconds

            explicit Queue(size_t capacity)
                : capacity_(capacity) { }

            bool push(T item) { return try_push(item); }

            bool pop(T& item) { return try_pop(item); }

            bool try_push(const T& item, Timeout timeout = std::nullopt) {
                std::unique_lock lock(mutex_);

                if (timeout.has_value()) {
                    if (!producer_.wait_for(lock, std::chrono::duration<double>(timeout.value()), [this] {
                            return !unsafe_full();
                        })) {
                        return false; // timeout
                    }
                } else {
                    producer_.wait(lock, [this] { return !unsafe_full(); });
                }

                queue_.push_back(item);
                consumer_.notify_one();
                return true;
            }

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

                item = std::move(queue_.front());
                queue_.pop_front();
                producer_.notify_one();
                return true;
            }

            void clear() {
                std::lock_guard lock(mutex_);
                std::deque<T>   empty;
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

            inline bool unsafe_full() const { return capacity_ <= queue_.size(); }

            inline bool unsafe_empty() const { return queue_.empty(); }

            size_t          capacity_;
            std::dequeue<T> queue_;

            mutable Mutex     mutex_;
            ConditionVariable producer_;
            ConditionVariable consumer_;
    };

    extern template class Queue<int>;
    extern template class Queue<uint>;

} // namespace corekit