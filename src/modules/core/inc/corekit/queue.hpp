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
                std::lock_guard<Mutex> lock(mutex);
                
                if(full()) {
                    return false;
                }

                queue.emplace_back(item);
                return true;
            }

            bool try_pop(T& item) {
                std::lock_guard<Mutex> lock(mutex);
                
                if (empty()) {
                    return false;
                }

                item = queue.front();
                queue.pop_front();
                return true;
            }

            void clear() {
                std::lock_guard<Mutex> lock(mutex);
                queue.clear();
            }

            bool empty() const {
                return queue.empty();
            }

            bool full() const {
                return capacity <= queue.size();
            }

            size_t size() const {
                return queue.size();
            }

            const size_t capacity;

           private:
            std::deque<T> queue;
            mutable Mutex mutex;
        };

};      // namespace corekit
