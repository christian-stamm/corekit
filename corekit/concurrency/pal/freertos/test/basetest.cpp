
#include <stdio.h>

#include "FreeRTOS.h"
#include "task.h"

extern "C" void vApplicationMallocFailedHook(void) {
    // Fail loudly during tests.
    __builtin_trap();
}

extern "C" void vApplicationStackOverflowHook(TaskHandle_t xTask,
                                              char        *pcTaskName) {
    // Fail loudly during tests.
    __builtin_trap();
}

extern "C" void vApplicationDaemonTaskStartupHook(void) {
    // Nothing required for tests.
}

static void task1(void *arg) {
    (void)arg;

    for (;;) {
        printf("Task 1 running\n");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

static void task2(void *arg) {
    (void)arg;

    for (;;) {
        printf("Task 2 running\n");
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

int main(void) {
    xTaskCreate(task1, "Task1", configMINIMAL_STACK_SIZE, NULL, 1, NULL);
    xTaskCreate(task2, "Task2", configMINIMAL_STACK_SIZE, NULL, 1, NULL);

    vTaskStartScheduler();

    /* Should never reach here if the scheduler starts successfully. */
    for (;;);
}