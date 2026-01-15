#pragma once
#include <deque>
#include <limits>
#include <mutex>

#include "corekit/system/concurrency/mutex.hpp"

namespace corekit {
    namespace system {
        namespace concurrency {

            template <typename T>
            class SafeQueue {
               public:
                friend class Executor;

                SafeQueue(
                    const size_t& capacity = std::numeric_limits<size_t>::max())
                    : queue()
                    , mutex()
                    , capacity(capacity){};

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

               private:
                bool push(const T& item, bool force = false) {
                    std::lock_guard<Mutex> lock(mutex);
                    const bool             full = capacity <= queue.size();

                    if (full && !force) {
                        return false;
                    }

                    queue.push_back(item);
                    return true;
                }

                const size_t  capacity;
                std::deque<T> queue;
                mutable Mutex mutex;
            };

        };  // namespace concurrency
    };      // namespace system
};          // namespace corekit
