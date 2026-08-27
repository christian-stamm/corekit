#include <FreeRTOS.h>
#include <task.h>

#include "corekit/logger.hpp"

void vApplicationMallocFailedHook() {}
void vApplicationStackOverflowHook(TaskHandle_t xTask, char* pcTaskName) {}
void vApplicationIdleHook() {}
void vApplicationTickHook() {}

int main() {
    {
        corekit::Logger logger("Hello");
        logger.info() << "Hello, World!";
        logger.debug() << "Hello, World!";
        logger.warn() << "Hello, World!";
        logger.error() << "Hello, World!";
        logger.fatal() << "Hello, World!";
    }

    return 0;
}