#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <semaphore>

#include "corekit/utils/assert.hpp"

namespace corekit {
    namespace utils {

        using namespace corekit::utils;

        template <typename T>
        class Observable {
           public:
            using OptValue    = std::optional<T>;
            using Subscriber  = std::function<void(const T& value)>;
            using Subscribers = std::vector<Subscriber>;

            Observable(const OptValue&    val  = std::nullopt,
                       const Subscribers& subs = {}) noexcept
                : val_(val)
                , subs_(subs) {}

            Observable(const Observable<T>& other) {
                std::lock_guard lock(other.mtx_);
                val_  = other.val_;
                subs_ = other.subs_;
            }

            Observable(const Observable<T>&& other) {
                std::lock_guard lock(other.mtx_);
                val_  = std::move(other.val_);
                subs_ = std::move(other.subs_);
            }

            void set(const T& value) {
                {
                    std::lock_guard lock(mtx_);
                    val_ = value;
                }

                notify();
            }

            const T& get() const {
                corecheck(valid(), "Observable::get => value is not set.");

                std::lock_guard lock(mtx_);
                return val_.value();
            }

            void reset() {
                {
                    std::lock_guard lock(mtx_);
                    val_.reset();
                }
            }

            bool valid() const {
                std::lock_guard lock(mtx_);
                return val_.has_value();
            }

            void subscribe(Subscriber cb) {
                std::lock_guard lock(mtx_);
                subs_.push_back(std::move(cb));
            }

           private:
            void notify() const {
                corecheck(valid(), "Observable::notify => value is not set.");

                std::lock_guard lock(mtx_);
                for (const Subscriber& cb : subs_) {
                    cb(val_.value());
                }
            }

            mutable std::atomic<bool> doNotify_;
            mutable std::mutex        mtx_;
            OptValue                  val_;
            Subscribers               subs_;
        };

    };  // namespace utils
};      // namespace corekit
