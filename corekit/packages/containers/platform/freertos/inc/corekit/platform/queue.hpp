#pragma once

#include <FreeRTOS.h>
#include <queue.h>

#include <array>
#include <type_traits>
#include <utility>

namespace corekit::platform {

    template <typename T>
    class Queue {
        // static_assert(std::is_trivially_copyable_v<T>,
        //               "Queue<T> uses FreeRTOS byte-copy queues; T must be "
        //               "trivially copyable.");

       protected:
        struct Opset {
            virtual inline BaseType_t send(QueueHandle_t q,
                                           const T&      item,
                                           TickType_t    wait = portMAX_DELAY) {
                return xQueueSendToBack(q, &item, wait);
            }

            virtual inline BaseType_t receive(QueueHandle_t q,
                                              T&            item,
                                              TickType_t wait = portMAX_DELAY) {
                return xQueueReceive(q, &item, wait);
            }
        };

        using CoreSet = Opset;

        struct IsrSet : public Opset {
            virtual inline BaseType_t send(QueueHandle_t q,
                                           const T&      item,
                                           TickType_t) override {
                return xQueueSendToBackFromISR(q, &item, nullptr);
            }

            virtual inline BaseType_t receive(QueueHandle_t q,
                                              T&            item,
                                              TickType_t) override {
                return xQueueReceiveFromISR(q, &item, nullptr);
            }
        };

       public:
        explicit Queue(size_t capacity,
                       Opset  cset = CoreSet(),
                       Opset  iset = IsrSet())
            : queue_(xQueueCreate(capacity, sizeof(T)))
            , core_set_(cset)
            , isr_set_(iset) {}

        bool push(T item, bool wait = true) {
            return push(xPortIsInsideInterrupt() ? isr_set_ : core_set_,
                        item,
                        wait ? portMAX_DELAY : 0);
        }

        bool pop(T& item, bool wait = true) {
            return pop(xPortIsInsideInterrupt() ? isr_set_ : core_set_,
                       item,
                       wait ? portMAX_DELAY : 0);
        }

        void clear() {
            xQueueReset(queue_);
        }

        bool empty() const {
            return size() == 0;
        }

        size_t size() const {
            return uxQueueGetQueueLength(queue_) -
                   uxQueueSpacesAvailable(queue_);
        }

       private:
        bool push(Opset& op, const T& item, TickType_t wait) {
            return op.send(queue_, item, wait) == pdTRUE;
        }

        bool pop(Opset& op, T& item, TickType_t wait) {
            return op.receive(queue_, item, wait) == pdTRUE;
        }

        Opset         core_set_;
        Opset         isr_set_;
        QueueHandle_t queue_;
    };

    extern template class Queue<int>;
    extern template class Queue<uint>;

}  // namespace corekit::platform