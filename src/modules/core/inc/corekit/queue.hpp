#pragma once
#include <deque>
#include <limits>

#include "corekit/mutex.hpp"

namespace corekit {

        template <typename T>
        class SafeQueue {
           public:

            SafeQueue(
                const size_t& capacity = std::numeric_limits<size_t>::max())
                : capacity(capacity){};

            SafeQueue(const SafeQueue&)             = delete;
            SafeQueue(const SafeQueue&&)            = delete;
            SafeQueue& operator=(const SafeQueue&)  = delete;
            SafeQueue& operator=(const SafeQueue&&) = delete;

            ~SafeQueue() = default;

            bool try_push(const T& item) {
                return this->push(item, false);
            }

            bool try_pop(T& item) {
                std::lock_guard<Mutex> lock(mutex);
                if (queue.empty()) {
                    return false;
                }

                item = queue.front();
                queue.pop_front();
                return true;
            }

            size_t size() const {
                std::lock_guard<Mutex> lock(mutex);
                return queue.size();
            }

            void clear() {
                std::lock_guard<Mutex> lock(mutex);
                queue.clear();
            }

            bool empty() const {
                std::lock_guard<Mutex> lock(mutex);
                return queue.empty();
            }

            bool full() const {
                std::lock_guard<Mutex> lock(mutex);
                return capacity <= queue.size();
            }

            const size_t capacity;

           private:
            bool push(const T& item, bool force = false) {
                if (full() && !force) {
                    return false;
                }

                std::lock_guard<Mutex> lock(mutex);
                queue.push_back(item);
                return true;
            }

            std::deque<T> queue;
            mutable Mutex mutex;
        };

};      // namespace corekit
