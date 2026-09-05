#pragma once
#include <memory>
#include <mutex>
#include <utility>

#include "corekit/mutex.hpp"
#include "corekit/result.hpp"

namespace corekit {

    template <typename T>
    class DoubleBuffer {
       public:
        using Ptr    = std::shared_ptr<DoubleBuffer<T>>;
        using Buffer = std::shared_ptr<T>;

        template <typename... Args>
        DoubleBuffer(Args&&... args)
            : ping(std::make_shared<T>(std::forward<Args>(args)...))
            , pong(std::make_shared<T>(std::forward<Args>(args)...)) {}

        Buffer read() const {
            std::lock_guard<Mutex> lock(mtx);
            return ping;
        }

        Buffer write() const {
            std::lock_guard<Mutex> lock(mtx);
            return pong;
        }

        VoidResult flip() {
            if (!ping || !pong) {
                return RuntimeError("DoubleBuffer is not initialized");
            }

            if (0 < current_reader_count()) {
                return RuntimeError(
                    "Cannot flip DoubleBuffer: read buffer is still in use");
            }

            if (0 < current_writer_count()) {
                return RuntimeError(
                    "Cannot flip DoubleBuffer: write buffer is still in use");
            }

            std::lock_guard<Mutex> lock(mtx);
            std::swap(ping, pong);
            return VoidResult();
        }

        size_t current_reader_count() const {
            return ping.use_count() -
                   1;  // Subtract one for the internal reference
        }

        size_t current_writer_count() const {
            return pong.use_count() -
                   1;  // Subtract one for the internal reference
        }

       private:
        Mutex  mtx;
        Buffer ping;
        Buffer pong;
    };

}  // namespace corekit