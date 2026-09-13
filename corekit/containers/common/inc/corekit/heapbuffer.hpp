#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "corekit/check.hpp"
#include "corekit/math.hpp"
#include "corekit/memory.hpp"

namespace corekit {

    template <typename T = std::byte>
    class HeapBuffer {
            template <typename>
            friend class Buffer;

            static_assert( //
                std::is_trivially_constructible_v<std::remove_const_t<T>>,
                "Buffer type must be trivially constructible");
            static_assert( //
                std::is_trivially_copyable_v<std::remove_const_t<T>>,
                "Buffer type must be trivially copyable");
            static_assert( //
                !std::is_reference_v<T>,
                "Buffer type must not be a reference");
            static_assert( //
                !std::is_volatile_v<T>,
                "Buffer type must not be volatile");
            static_assert( //
                std::is_object_v<std::remove_const_t<T>>,
                "Buffer type must be an object type");

        public:

            using value_type = T;
            using Ptr        = std::shared_ptr<HeapBuffer<T>>;
            using Cookie     = std::shared_ptr<void>;

            HeapBuffer() noexcept
                : ctrl_(nullptr)
                , data_(nullptr)
                , size_(0) { }

            HeapBuffer(std::size_t count) {
                const std::size_t bytes = count * sizeof(T);

                core::check( //
                    bytes <= std::numeric_limits<std::size_t>::max(),
                    RuntimeError("buffer size overflow"));

                core::check( //
                    0 < count,
                    RuntimeError("buffer size must be greater than zero"));

                void* ptr = mem::malloc(bytes, alignof(T));

                core::check( //
                    ptr != nullptr,
                    RuntimeError("buffer allocation failed"));

                this->ctrl_ = Cookie(ptr, [](void* p) { mem::free(p); });
                this->data_ = reinterpret_cast<T*>(ptr);
                this->size_ = count;
            }

            HeapBuffer(std::size_t count, const T& value)
                : HeapBuffer(count) {
                this->fill(value);
            }

            template <typename U>
            requires std::is_convertible_v<U*, T*>
            HeapBuffer(const HeapBuffer<U>& other) noexcept
                : ctrl_(other.ctrl_)
                , data_(other.data_)
                , size_(other.size_) { }

            HeapBuffer(HeapBuffer&& other) noexcept
                : ctrl_(std::move(other.ctrl_))
                , data_(std::exchange(other.data_, nullptr))
                , size_(std::exchange(other.size_, 0)) { }

            HeapBuffer& operator=(const HeapBuffer& other) = default;

            HeapBuffer& operator=(HeapBuffer&& other) noexcept {
                if (this == &other) { return *this; }

                ctrl_ = std::move(other.ctrl_);
                data_ = std::exchange(other.data_, nullptr);
                size_ = std::exchange(other.size_, 0);

                return *this;
            }

            T& front() noexcept { return at(0); }

            const T& front() const noexcept { return at(0); }

            T& back() noexcept { return at(-1); }

            const T& back() const noexcept { return at(-1); }

            T& at(int64_t index) noexcept { return data_[math::wrap(index, size_)]; }

            const T& at(int64_t index) const noexcept { return data_[math::wrap(index, size_)]; }

            [[nodiscard]]
            T* data() noexcept {
                return data_;
            }

            [[nodiscard]]
            const T* data() const noexcept {
                return data_;
            }

            [[nodiscard]]
            std::size_t size() const noexcept {
                return size_;
            }

            [[nodiscard]]
            std::size_t size_bytes() const noexcept {
                return size_ * sizeof(T);
            }

            [[nodiscard]]
            bool empty() const noexcept {
                return size_ == 0;
            }

            [[nodiscard]]
            bool unique() const noexcept {
                return ctrl_.unique();
            }

            [[nodiscard]]
            std::size_t instances() const noexcept {
                return ctrl_.use_count();
            }

            T& operator[](int64_t index) noexcept { return at(index); }

            const T& operator[](int64_t index) const noexcept { return at(index); }

            T* begin() noexcept { return data_; }

            T* end() noexcept { return size_ ? data_ + size_ : data_; }

            const T* begin() const noexcept { return data_; }

            const T* end() const noexcept { return size_ ? data_ + size_ : data_; }

            [[nodiscard]]
            HeapBuffer<T> view(std::size_t offset, std::size_t count) {
                core::check(offset <= size_, RuntimeError("buffer offset out of bounds"));
                core::check(count <= size_ - offset, RuntimeError("buffer length out of bounds"));

                HeapBuffer<T> result;
                result.ctrl_ = ctrl_;
                result.data_ = size_ ? data_ + offset : data_;
                result.size_ = count;
                return result;
            }

            [[nodiscard]]
            HeapBuffer<const T> view(std::size_t offset, std::size_t count) const {
                core::check(offset <= size_, RuntimeError("buffer offset out of bounds"));
                core::check(count <= size_ - offset, RuntimeError("buffer length out of bounds"));

                HeapBuffer<const T> result;
                result.ctrl_ = ctrl_;
                result.data_ = size_ ? data_ + offset : data_;
                result.size_ = count;
                return result;
            }

            template <typename U>
            [[nodiscard]]
            auto cast() {
                using R = std::conditional_t<std::is_const_v<T>, const U, U>;

                core::check(
                    size_bytes() % sizeof(U) == 0,
                    RuntimeError("buffer size is incompatible with target type"));

                core::check(
                    reinterpret_cast<uintptr_t>(data_) % alignof(U) == 0,
                    RuntimeError("buffer is not aligned for target type"));

                HeapBuffer<R> result;
                result.ctrl_ = ctrl_;
                result.data_ = reinterpret_cast<R*>(data_);
                result.size_ = size_bytes() / sizeof(U);
                return result;
            }

            template <typename U>
            [[nodiscard]]
            HeapBuffer<const U> cast() const {
                core::check(
                    size_bytes() % sizeof(U) == 0,
                    RuntimeError("buffer size is incompatible with target type"));

                core::check(
                    reinterpret_cast<uintptr_t>(data_) % alignof(U) == 0,
                    RuntimeError("buffer is not aligned for target type"));

                HeapBuffer<const U> result;
                result.ctrl_ = ctrl_;
                result.data_ = reinterpret_cast<const U*>(data_);
                result.size_ = size_bytes() / sizeof(U);
                return result;
            }

            HeapBuffer<const T> as_const() const noexcept {
                HeapBuffer<const T> result;
                result.ctrl_ = ctrl_;
                result.data_ = data_;
                result.size_ = size_;
                return result;
            }

            void fill(const T& value) noexcept { std::fill(begin(), end(), value); }

        private:

            Cookie      ctrl_;
            T*          data_;
            std::size_t size_;
    };

} // namespace corekit