#pragma once

#include <FreeRTOS.h>
#include <queue.h>

#include <optional>
#include <type_traits>

#include "corekit/check.hpp"

namespace corekit::platform {

    template <typename T>
    class Queue {
            static_assert(std::is_trivially_copyable_v<T>, "Queue<T> requires T to be trivially copyable");

        public:

            using Timeout = std::optional<double>; // seconds

            explicit Queue(size_t capacity)
                : queue_(xQueueCreate(capacity, sizeof(T))) {
                corecheck(queue_ != nullptr, RuntimeError("Queue::Queue() failed to create queue"));
            }

            ~Queue() {
                if (queue_) { vQueueDelete(queue_); }
            }

            Queue(const Queue&)            = delete;
            Queue(Queue&&)                 = delete;
            Queue& operator=(const Queue&) = delete;
            Queue& operator=(Queue&&)      = delete;

            bool push(const T item) { return try_push(item); }

            bool pop(T& item) { return try_pop(item); }

            virtual bool try_push(const T& item, Timeout seconds = std::nullopt) {
                BaseType_t result = pdFALSE;

                if (xPortIsInsideInterrupt()) {
                    BaseType_t wake = pdFALSE;
                    result          = xQueueSendToBackFromISR(queue_, &item, &wake);
                    portYIELD_FROM_ISR(wake);
                } else {
                    if (seconds.has_value()) {
                        result = xQueueSendToBack(queue_, &item, pdMS_TO_TICKS(1e3 * seconds.value()));
                    } else {
                        result = xQueueSendToBack(queue_, &item, portMAX_DELAY);
                    }
                }

                return result == pdTRUE;
            }

            virtual bool try_pop(T& item, Timeout seconds = std::nullopt) {
                BaseType_t result = pdFALSE;

                if (xPortIsInsideInterrupt()) {
                    BaseType_t wake = pdFALSE;
                    result          = xQueueReceiveFromISR(queue_, &item, &wake);
                    portYIELD_FROM_ISR(wake);
                } else {
                    if (seconds.has_value()) {
                        result = xQueueReceive(queue_, &item, pdMS_TO_TICKS(1e3 * seconds.value()));
                    } else {
                        result = xQueueReceive(queue_, &item, portMAX_DELAY);
                    }
                }

                return result == pdTRUE;
            }

            inline void clear() { xQueueReset(queue_); }

            inline bool empty() const { return size() == 0; }

            inline size_t size() const { return uxQueueGetQueueLength(queue_); }

            inline bool full() const { return uxQueueSpacesAvailable(queue_) == 0; }

        protected:

            QueueHandle_t queue_;
    };

    extern template class Queue<int>;
    extern template class Queue<uint>;

} // namespace corekit::platform