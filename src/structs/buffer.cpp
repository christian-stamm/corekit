#include "corekit/structs/buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "corekit/core.hpp"

namespace corekit {
    namespace structs {

        template <typename T>
        Buffer<T>::Buffer(size_t size, T value, bool align)
            : Buffer(request(size * sizeof(T), align),
                     0,
                     size,
                     std::make_shared<bool>(true)) {
            this->fill(value);
        }

        template <typename T>
        Buffer<T>::Buffer(const std::span<T>& span)
            : Buffer(span.data(),
                     0,
                     span.size(),
                     std::make_shared<bool>(false)) {}

        template <typename T>
        Buffer<T>::Buffer(const std::vector<T>& buffer)
            : Buffer(const_cast<T*>(buffer.data()),
                     0,
                     buffer.size(),
                     std::make_shared<bool>(false)) {}

        template <typename T>
        Buffer<T>::Buffer(const Buffer<T>& other)
            : Buffer(other.base, other.lower, other.upper, other.owners) {}

        template <typename T>
        Buffer<T>::Buffer(const Buffer<T>&& other)
            : Buffer(other.base,
                     other.lower,
                     other.upper,
                     std::move(other.owners)) {}

        template <typename T>
        Buffer<T>::Buffer(T*           base,
                          const size_t lower,
                          const size_t upper,
                          const Owner& owners)
            : std::span<T>(base + lower, upper - lower)
            , base(base)
            , lower(lower)
            , upper(upper)
            , owners(owners) {
            if constexpr ((!std::is_integral_v<T> || std::is_signed_v<T>) &&
                          !std::is_same_v<T, uintptr_t>) {
                throw std::runtime_error(
                    "Buffer only supports unsigned integral data types or "
                    "uintptr_t.");
            }
        }

        template <typename T>
        Buffer<T>::~Buffer() {
            if (isOwner()) {
                release(base);
            }
        }

        template <typename T>
        T& Buffer<T>::operator[](int index) const {
            return std::span<T>::operator[](math::wrap(index, this->size()));
        }

        template <typename T>
        Buffer<T>& Buffer<T>::operator=(const Buffer<T>& other) {
            if (this != &other) {
                std::span<T>::operator=(other);
                base   = other.base;
                lower  = other.lower;
                upper  = other.upper;
                owners = other.owners;
            }

            return *this;
        }

        template <typename T>
        Buffer<T>& Buffer<T>::operator=(const Buffer<T>&& other) {
            if (this != &other) {
                std::span<T>::operator=(other);
                base   = other.base;
                lower  = other.lower;
                upper  = other.upper;
                owners = std::move(other.owners);
            }

            return *this;
        }

        template <typename T>
        bool Buffer<T>::operator==(const Buffer<T>& other) const {
            if (this == &other) {
                return true;
            }

            return std::equal(this->begin(),
                              this->end(),
                              other.begin(),
                              other.end());
        }

        template <typename T>
        Buffer<T>& Buffer<T>::operator<<(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element <<= value;
            });
            return *this;
        }

        template <typename T>
        Buffer<T>& Buffer<T>::operator>>(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element >>= value;
            });
            return *this;
        }

        template <typename T>
        Buffer<T>& Buffer<T>::operator+(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element += value;
            });
            return *this;
        }

        template <typename T>
        Buffer<T>& Buffer<T>::operator-(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element -= value;
            });
            return *this;
        }

        template <typename T>
        Buffer<T>& Buffer<T>::operator%(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element %= value;
            });
            return *this;
        }

        template <typename T>
        Buffer<T>& Buffer<T>::operator&(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element &= value;
            });
            return *this;
        }

        template <typename T>
        Buffer<T>& Buffer<T>::fill(const T& value) {
            std::fill(this->begin(), this->end(), value);
            return *this;
        }

        template <typename T>
        Buffer<T>& Buffer<T>::iota(const T& base) {
            std::iota(this->begin(), this->end(), base);
            return *this;
        }

        template <typename T>
        void Buffer<T>::copyTo(Buffer<T>& target) const {
            if (this->size() != target.size()) {
                throw std::runtime_error("Buffer copyTo size mismatch.");
            }

            std::copy(this->begin(), this->end(), target.begin());
        }

        template <typename T>
        Buffer<T> Buffer<T>::extend(size_t shift, size_t length) const {
            const size_t availableCapacity = (upper - lower);
            const size_t requestedCapacity = (shift + length);
            const bool   overflow = availableCapacity < requestedCapacity;

            if (overflow) {
                Buffer<T> resized(requestedCapacity, 0);
                Buffer<T> slice = this->subset(0, this->size());

                this->copyTo(slice);
                return resized;
            }

            return Buffer<T>(base, lower, lower + requestedCapacity, owners);
        }

        template <typename T>
        Buffer<T> Buffer<T>::subset(size_t shift, size_t length) const {
            const size_t availableCapacity = (upper - lower);
            const size_t requestedCapacity = (shift + length);
            const bool   overflow = availableCapacity < requestedCapacity;

            if (overflow) {
                throw std::runtime_error(std::format(
                    "Overflow: Requested Capacity={}, Available Capacity={}",
                    requestedCapacity,
                    availableCapacity));
            }

            return Buffer<T>(base,
                             lower + shift,
                             lower + requestedCapacity,
                             owners);
        }

        template <typename T>
        Buffer<T>::List Buffer<T>::split(size_t numSplits) const {
            const size_t splitSize = this->size() / numSplits;

            size_t processed = 0;
            size_t remaining = this->size();

            Buffer<T>::List splits(numSplits);

            for (size_t part = 0; part < numSplits - 1; part++) {
                splits[part] = subset(processed, splitSize);

                processed += splitSize;
                remaining -= splitSize;
            }

            splits[numSplits - 1] = subset(processed, remaining);

            return splits;
        }

        template <typename T>
        bool Buffer<T>::isAligned() const {
            const uintptr_t address = reinterpret_cast<uintptr_t>(base + lower);
            const size_t    typeSize = sizeof(T);

            return (address % typeSize) == 0;
        }

        template <typename T>
        bool Buffer<T>::isOwner() const {
            return *owners;
        }

        template <typename T>
        T* Buffer<T>::request(size_t size, bool align) {
            align &= std::has_single_bit(size);
            void* ptr = (align ? aligned_alloc(size, size) : std::malloc(size));

            if (ptr == nullptr) {
                throw std::bad_alloc();
            }

            return static_cast<T*>(ptr);
        }

        template <typename T>
        void Buffer<T>::release(const T* ptr) {
            if (!isOwner()) {
                throw std::runtime_error("Memory release on non-owner Buffer.");
            }

            if (owners.use_count() <= 1) {
                free(const_cast<T*>(ptr));
            }
        }

        template class Buffer<uint8_t>;
        template class Buffer<uint16_t>;
        template class Buffer<uint32_t>;

    };  // namespace structs
};  // namespace corekit
