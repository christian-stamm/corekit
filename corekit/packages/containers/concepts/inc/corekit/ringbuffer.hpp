#pragma once
#include <array>
#include <memory>
#include <mutex>

#include "corekit/math.hpp"
#include "corekit/mutex.hpp"

namespace corekit {

    template <typename T, size_t N>
    class RingBuffer {
        public:

            using Ptr = std::shared_ptr<RingBuffer<T, N>>;

            RingBuffer()
                : count(0)
                , head(0)
                , tail(0) { }

            bool push(const T& item) {
                std::lock_guard<Mutex> lock(mtx);

                if (count == N) {
                    return false; // Buffer is full
                }

                buffer[head] = item;
                head         = (head + 1) % N;
                ++count;

                return true;
            }

            bool at(int index, T& item) const {
                std::lock_guard<Mutex> lock(mtx);

                if (abs(index) >= count) {
                    return false; // Index out of bounds
                }

                index = math::wrap(tail + index, N);
                item  = buffer[index];

                return true;
            }

            bool erase(const T& item) {
                std::lock_guard<Mutex> lock(mtx);

                for (size_t i = 0; i < count; ++i) {
                    const size_t pos = (tail + i) % N;

                    if (buffer[pos] == item) {
                        for (size_t j = i; j + 1 < count; ++j) {
                            const size_t from = (tail + j + 1) % N;
                            const size_t to   = (tail + j) % N;
                            buffer[to]        = std::move(buffer[from]);
                        }

                        head = (head + N - 1) % N;
                        --count;
                        return true;
                    }
                }

                return false;
            }

            bool pop(T& item) {
                std::lock_guard<Mutex> lock(mtx);

                if (count == 0) {
                    return false; // Buffer is empty
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

            void clear() {
                std::lock_guard<Mutex> lock(mtx);
                count = 0;
                head  = 0;
                tail  = 0;
            }

        private:

            mutable Mutex mtx;

            size_t count;
            size_t head;
            size_t tail;

            std::array<T, N> buffer;
    };

} // namespace corekit