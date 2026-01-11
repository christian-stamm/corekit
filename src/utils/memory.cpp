#include "corekit/utils/memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "corekit/utils/math.hpp"

namespace corekit {
    namespace utils {

        using namespace corekit::math;

        template <typename T>
        Memory<T>::Memory(size_t size, T value, bool align)
            : Memory(request(size * sizeof(T), align),
                     0,
                     size,
                     std::make_shared<bool>(true)) {
            this->fill(value);
        }

        template <typename T>
        Memory<T>::Memory(const std::span<T>& span)
            : Memory(span.data(),
                     0,
                     span.size(),
                     std::make_shared<bool>(false)) {}

        template <typename T>
        Memory<T>::Memory(const std::vector<T>& memory)
            : Memory(const_cast<T*>(memory.data()),
                     0,
                     memory.size(),
                     std::make_shared<bool>(false)) {}

        template <typename T>
        Memory<T>::Memory(const Memory<T>& other)
            : Memory(other.base, other.lower, other.upper, other.owners) {}

        template <typename T>
        Memory<T>::Memory(const Memory<T>&& other)
            : Memory(other.base,
                     other.lower,
                     other.upper,
                     std::move(other.owners)) {}

        template <typename T>
        Memory<T>::Memory(T*           base,
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
                    "Memory only supports unsigned integral data types or "
                    "uintptr_t.");
            }
        }

        template <typename T>
        Memory<T>::~Memory() {
            if (isOwner()) {
                release(base);
            }
        }

        template <typename T>
        T& Memory<T>::operator[](int index) const {
            return std::span<T>::operator[](ops::wrap(index, this->size()));
        }

        template <typename T>
        Memory<T>& Memory<T>::operator=(const Memory<T>& other) {
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
        Memory<T>& Memory<T>::operator=(const Memory<T>&& other) {
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
        bool Memory<T>::operator==(const Memory<T>& other) const {
            if (this == &other) {
                return true;
            }

            return std::equal(this->begin(),
                              this->end(),
                              other.begin(),
                              other.end());
        }

        template <typename T>
        Memory<T>& Memory<T>::operator<<(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element <<= value;
            });
            return *this;
        }

        template <typename T>
        Memory<T>& Memory<T>::operator>>(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element >>= value;
            });
            return *this;
        }

        template <typename T>
        Memory<T>& Memory<T>::operator+(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element += value;
            });
            return *this;
        }

        template <typename T>
        Memory<T>& Memory<T>::operator-(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element -= value;
            });
            return *this;
        }

        template <typename T>
        Memory<T>& Memory<T>::operator%(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element %= value;
            });
            return *this;
        }

        template <typename T>
        Memory<T>& Memory<T>::operator&(uint value) {
            std::for_each(this->begin(), this->end(), [value](T& element) {
                element &= value;
            });
            return *this;
        }

        template <typename T>
        Memory<T>& Memory<T>::fill(const T& value) {
            std::fill(this->begin(), this->end(), value);
            return *this;
        }

        template <typename T>
        Memory<T>& Memory<T>::iota(const T& base) {
            std::iota(this->begin(), this->end(), base);
            return *this;
        }

        template <typename T>
        void Memory<T>::copyTo(Memory<T>& target) const {
            if (this->size() != target.size()) {
                throw std::runtime_error("Memory copyTo size mismatch.");
            }

            std::copy(this->begin(), this->end(), target.begin());
        }

        template <typename T>
        Memory<T> Memory<T>::extend(size_t shift, size_t length) const {
            const size_t availableCapacity = (upper - lower);
            const size_t requestedCapacity = (shift + length);
            const bool   overflow = availableCapacity < requestedCapacity;

            if (overflow) {
                Memory<T> resized(requestedCapacity, 0);
                Memory<T> slice = this->subset(0, this->size());

                this->copyTo(slice);
                return resized;
            }

            return Memory<T>(base, lower, lower + requestedCapacity, owners);
        }

        template <typename T>
        Memory<T> Memory<T>::subset(size_t shift, size_t length) const {
            const size_t availableCapacity = (upper - lower);
            const size_t requestedCapacity = (shift + length);
            const bool   overflow = availableCapacity < requestedCapacity;

            if (overflow) {
                throw std::runtime_error(std::format(
                    "Overflow: Requested Capacity={}, Available Capacity={}",
                    requestedCapacity,
                    availableCapacity));
            }

            return Memory<T>(base,
                             lower + shift,
                             lower + requestedCapacity,
                             owners);
        }

        template <typename T>
        Memory<T>::List Memory<T>::split(size_t numSplits) const {
            const size_t splitSize =
                std::max<size_t>(1, this->size() / numSplits);

            if (this->size() < (splitSize * numSplits)) {
                throw std::runtime_error(
                    "Memory split size exceeds total size.");
            }

            size_t processed = 0;
            size_t remaining = this->size();

            Memory<T>::List splits(numSplits);

            for (size_t part = 0; part < numSplits - 1; part++) {
                splits[part] = subset(processed, splitSize);

                processed += splitSize;
                remaining -= splitSize;

                if (remaining == 0) {
                    break;
                }
            }

            if (0 < remaining) {
                splits[numSplits - 1] = subset(processed, remaining);
            }

            return splits;
        }

        template <typename T>
        bool Memory<T>::isAligned() const {
            const uintptr_t address = reinterpret_cast<uintptr_t>(base + lower);
            const size_t    typeSize = sizeof(T);

            return (address % typeSize) == 0;
        }

        template <typename T>
        size_t Memory<T>::sizeBytes() const {
            return this->size() * sizeof(T);
        }

        template <typename T>
        bool Memory<T>::isOwner() const {
            return *owners;
        }

        template <typename T>
        T* Memory<T>::request(size_t size, bool align) {
            align &= std::has_single_bit(size);
            void* ptr = (align ? aligned_alloc(size, size) : std::malloc(size));

            if (ptr == nullptr) {
                throw std::bad_alloc();
            }

            return static_cast<T*>(ptr);
        }

        template <typename T>
        void Memory<T>::release(const T* ptr) {
            if (!isOwner()) {
                throw std::runtime_error("Memory release on non-owner Memory.");
            }

            if (owners.use_count() <= 1) {
                free(const_cast<T*>(ptr));
            }
        }

        template class Memory<uint8_t>;
        template class Memory<uint16_t>;
        template class Memory<uint32_t>;
        template class Memory<uint64_t>;

    };  // namespace utils
};  // namespace corekit
