#pragma once

#include <functional>
#include <mutex>
#include <utility>

#include "corekit/mutex.hpp"

namespace corekit {

    template <typename T>
    class Synchronized {
        public:

            template <typename... Args>
            explicit Synchronized(std::in_place_t = std::in_place, Args&&... args)
                : data_(std::forward<Args>(args)...) { }

            Synchronized(const Synchronized&)            = delete;
            Synchronized(Synchronized&&)                 = delete;
            Synchronized& operator=(const Synchronized&) = delete;
            Synchronized& operator=(Synchronized&&)      = delete;

            template <typename Func>
            auto with_read(Func&& fn) const {
                std::scoped_lock<Mutex> lock(mutex_);
                return std::invoke(std::forward<Func>(fn), std::as_const(data_));
            }

            template <typename Func>
            bool try_with_read(Func&& fn) const {
                std::unique_lock<Mutex> lock(mutex_, std::try_to_lock);

                if (!lock.owns_lock()) { return false; }

                std::invoke(std::forward<Func>(fn), std::as_const(data_));

                return true;
            }

            template <typename Func>
            auto with_write(Func&& fn) {
                std::scoped_lock<Mutex> lock(mutex_);
                return std::invoke(std::forward<Func>(fn), data_);
            }

            template <typename Func>
            bool try_with_write(Func&& fn) {
                std::unique_lock<Mutex> lock(mutex_, std::try_to_lock);

                if (!lock.owns_lock()) { return false; }

                std::invoke(std::forward<Func>(fn), data_);
                return true;
            }

        private:

            mutable Mutex mutex_;
            T             data_;
    };

} // namespace corekit