#pragma once

#include <FreeRTOS.h>
#include <queue.h>

#include "corekit/queue.hpp"

namespace corekit::platform {

    template <typename T>
    class Stack : public Queue<T> {
       public:
        struct CoreSet : public Queue<T>::Opset {
            virtual inline BaseType_t send(QueueHandle_t q,
                                           T             item,
                                           TickType_t    wait = portMAX_DELAY) {
                return xQueueSendToFront(q, &item, wait);
            }

            virtual inline BaseType_t receive(QueueHandle_t q,
                                              T&            item,
                                              TickType_t wait = portMAX_DELAY) {
                return xQueueReceive(q, &item, wait);
            }
        };

        struct IsrSet : public Queue<T>::Opset {
            virtual inline BaseType_t send(QueueHandle_t q,
                                           T             item,
                                           TickType_t) override {
                return xQueueSendToFrontFromISR(q, &item, nullptr);
            }

            virtual inline BaseType_t receive(QueueHandle_t q,
                                              T&            item,
                                              TickType_t) override {
                return xQueueReceiveFromISR(q, &item, nullptr);
            }
        };

        explicit Stack(size_t capacity)
            : Queue<T>(capacity, Stack::CoreSet(), Stack::IsrSet()) {}
    };

    extern template class Stack<int>;
    extern template class Stack<uint>;

}  // namespace corekit::platform