#pragma once
#include <array>
#include <memory>
#include <mutex>

#include "corekit/mutex.hpp"

namespace corekit {

    template <typename T, size_t N>
    class RingBuffer {
       public:
        using Ptr = std::shared_ptr<RingBuffer<T, N>>;

        RingBuffer() : count(0), head(0), tail(0) {}

        bool push(const T& item) {
            std::lock_guard<Mutex> lock(mtx);

            if (count == N) {
                return false;  // Buffer is full
            }

            buffer[head] = item;
            head         = (head + 1) % N;
            ++count;

            return true;
        }

        bool pop(T& item) {
            std::lock_guard<Mutex> lock(mtx);

            if (count == 0) {
                return false;  // Buffer is empty
            }

            item = buffer[tail];
            tail = (tail + 1) % N;
            --count;

            return true;
        }

        bool empty() const {
            std::lock_guard<Mutex> lock(mtx);
            return count == 0;
        }

        bool full() const {
            std::lock_guard<Mutex> lock(mtx);
            return count == N;
        }

       private:
        Mutex  mtx;
        size_t count;
        size_t head;
        size_t tail;

        std::array<T, N> buffer;
    };

}  // namespace corekit