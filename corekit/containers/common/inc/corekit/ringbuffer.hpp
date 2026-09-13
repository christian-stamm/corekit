#pragma once
#include <array>
#include <memory>
#include <mutex>

#include "corekit/check.hpp"
#include "corekit/math.hpp"
#include "corekit/mutex.hpp"

namespace corekit {

    template <typename T, size_t N>
    class Ringbuffer {
        public:

            using Ptr = std::shared_ptr<Ringbuffer<T, N>>;

            Ringbuffer()
                : count_(0)
                , head_(0)
                , tail_(0) { }

            bool at(int index, T& item) const {
                std::lock_guard<Mutex> lock(mtx);
                core::check(abs(index) < count_, OutOfRangeError("Index out of bounds"));

                index = math::wrap(tail_ + index, N);
                item  = buffer_[index];

                return core::ok();
            }

            bool rotate(int offset) {
                std::lock_guard<Mutex> lock(mtx);
                core::check(abs(offset) < count_, OutOfRangeError("Offset out of bounds"));
                tail_ = math::wrap(tail_ + offset, N);
                head_ = math::wrap(head_ + offset, N);
                return core::ok();
            }

            bool push(const T& item) {
                std::lock_guard<Mutex> lock(mtx);

                if (count_ == N) {
                    return false; // buffer_ is full
                }

                buffer_[head_] = item;
                head_          = math::wrap(head_ + 1, N);
                ++count_;

                return true;
            }

            bool pop(T& item) {
                std::lock_guard<Mutex> lock(mtx);

                if (count_ == 0) {
                    return false; // buffer_ is empty
                }

                item  = buffer_[tail_];
                tail_ = math::wrap(tail_ + 1, N);
                --count_;

                return true;
            }

            bool empty() const {
                std::lock_guard<Mutex> lock(mtx);
                return count_ == 0;
            }

            bool full() const {
                std::lock_guard<Mutex> lock(mtx);
                return count_ == N;
            }

            size_t size() const {
                std::lock_guard<Mutex> lock(mtx);
                return count_;
            }

            size_t capacity() const { return N; }

            void clear() {
                std::lock_guard<Mutex> lock(mtx);
                count_ = 0;
                head_  = 0;
                tail_  = 0;
            }

        private:

            mutable Mutex mtx;

            size_t count_;
            size_t head_;
            size_t tail_;

            std::array<T, N> buffer_;
    };

} // namespace corekit