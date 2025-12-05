#pragma once
//
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace corekit {
namespace structs {

    template <typename T> class Buffer : public std::span<T> {
        using Owner = std::shared_ptr<bool>;

      public:
        template <typename N> friend class Buffer;
        using Ptr  = std::shared_ptr<Buffer>;
        using List = std::vector<Buffer<T>>;

        Buffer(size_t size = 0, T value = 0, bool align = false);
        Buffer(const std::span<T>& span);
        Buffer(const std::vector<T>& buffer);
        Buffer(const Buffer<T>& other);
        Buffer(const Buffer<T>&& other);
        Buffer(const Buffer<T>::Ptr& other);
        ~Buffer();

        Buffer<T>& operator=(const Buffer<T>& other);
        Buffer<T>& operator=(const Buffer<T>&& other);

        std::span<T> subspan() = delete;

        T&   operator[](int index) const;
        bool operator==(const Buffer<T>& other) const;

        Buffer<T>& operator<<(uint value);
        Buffer<T>& operator>>(uint value);
        Buffer<T>& operator+(uint value);
        Buffer<T>& operator-(uint value);
        Buffer<T>& operator%(uint value);
        Buffer<T>& operator&(uint value);

        friend std::ostream& operator<<(std::ostream& os, const Buffer<T>& obj)
        {
            const int dataWidth = std::numeric_limits<T>::digits;
            const int lineWidth = std::log10(obj.size()) + 1;

            os << "Buffer(" << std::endl;

            for (uint32_t line = 0; line < obj.size(); line++) {
                os << std::format(
                          "\t{:0{}}: 0b{:0{}b} / 0x{:0{}X} / {}", //
                          line, lineWidth, obj[line], dataWidth, obj[line], dataWidth / 4, obj[line])
                   << std::endl;
            }

            os << ")";
            return os;
        }

        template <typename N> Buffer<N> cast() const
        {
            if constexpr (std::is_same_v<T, N>) {
                return *this;
            }

            if (sizeof(T) % sizeof(N) != 0 && sizeof(N) % sizeof(T) != 0) {
                throw std::runtime_error("Incompatible Buffer cast types.");
            }

            if (this->sizeBytes() % sizeof(N) != 0) {
                throw std::runtime_error("Buffer size is not compatible with cast type.");
            }

            const size_t newLower = (lower * sizeof(T)) / (1.0 * sizeof(N));
            const size_t newUpper = (upper * sizeof(T)) / (1.0 * sizeof(N));

            return Buffer<N>(reinterpret_cast<N*>(base), newLower, newUpper, owners);
        }

        Buffer<T>  extend(size_t shift, size_t length) const;
        Buffer<T>  subset(size_t shift, size_t length) const;
        void       copyTo(Buffer<T>& target) const;
        List       split(size_t numSplits) const;
        Buffer<T>& fill(const T& value);
        Buffer<T>& iota(const T& base = 0);

        size_t sizeBytes() const;
        bool   isAligned() const;
        bool   isOwner() const;

      protected:
        Buffer(T* base, size_t lower, size_t upper, const Owner& owners);

      private:
        T*   request(size_t size, bool align);
        void release(const T* ptr);

        T*     base;
        size_t lower;
        size_t upper;
        Owner  owners;
    };

}; // namespace structs
}; // namespace corekit
