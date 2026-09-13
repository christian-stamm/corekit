#pragma once

#include "corekit/queue.hpp"

namespace corekit {

    template <typename T>
    class Stack : public Queue<T> {
        public:

            explicit Stack(size_t capacity)
                : Queue<T>(capacity) {
                core::check(this->queue_ != nullptr, RuntimeError("Stack::Stack() failed to create stack"));
            }

            virtual bool try_push(                                //
                const T&                   item,                  //
                typename Queue<T>::Timeout seconds = std::nullopt //
                ) override                                        //
            {
                BaseType_t result = pdFALSE;

                if (xPortIsInsideInterrupt()) {
                    BaseType_t wake = pdFALSE;
                    result          = xQueueSendToFrontFromISR(this->queue_, &item, &wake);
                    portYIELD_FROM_ISR(wake);
                } else {
                    if (seconds.has_value()) {
                        result = xQueueSendToFront(this->queue_, &item, pdMS_TO_TICKS(1e3 * seconds.value()));
                    } else {
                        result = xQueueSendToFront(this->queue_, &item, portMAX_DELAY);
                    }
                }

                return result == pdTRUE;
            }
    };

    extern template class Stack<int>;
    extern template class Stack<uint>;

} // namespace corekit