#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <utility>

#include "corekit/math.hpp"

namespace corekit {

    template <typename T = uint8_t>
    class Memory : public std::span<T> {
       public:
        using Ptr = std::shared_ptr<Memory>;

        Memory(size_t size, bool aligned = true)
            : Memory(mem_alloc(size, aligned), size) {
            std::fill(this->begin(), this->end(), 0x00);
        }

        Memory(const Memory&)            = delete;
        Memory& operator=(const Memory&) = delete;

        Ptr static request(size_t size, bool aligned = true) {
            return std::make_shared<Memory>(size, aligned);
        }

        bool isAligned() const {
            return isAligned<T>(base, std::span<T>::size());
        }

        template <typename P>
        static bool isAligned(P* ptr, size_t alignment) {
            return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
        }

        ~Memory() {
            mem_free();
        }

       private:
        Memory(T* base, size_t size) : std::span<T>(base, size), base(base) {}

        T* mem_alloc(size_t size, bool aligned) {
            size *= sizeof(T);
            aligned &= math::isPow2(size);
            return static_cast<T*>(aligned ? aligned_alloc(size, size)
                                           : malloc(size));
        }

        void mem_free() {
            if (base) {
                free(base);
            }
        }

        T* base;
    };

    template <typename T>
    class DoubleBuffer {
       public:
        using Ptr = std::shared_ptr<DoubleBuffer<T>>;

        DoubleBuffer(size_t size, bool aligned = false)
            : buffer_ping(Memory<T>::request(size, aligned))
            , buffer_pong(Memory<T>::request(size, aligned)) {}

        static Ptr request(size_t size, bool aligned = false) {
            return std::make_shared<DoubleBuffer<T>>(size, aligned);
        }

        Memory<T>::Ptr read() const {
            return buffer_ping;
        }

        Memory<T>::Ptr write() const {
            return buffer_pong;
        }

        void flip() {
            std::swap(buffer_ping, buffer_pong);
        }

       private:
        Memory<T>::Ptr buffer_ping;
        Memory<T>::Ptr buffer_pong;
    };

}  // namespace corekit