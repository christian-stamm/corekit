#include <FreeRTOS.h>
#include <task.h>

#include <format>
#include <iostream>

extern "C" {

void vApplicationStackOverflowHook(TaskHandle_t task, char* task_name) {
    taskDISABLE_INTERRUPTS();

    std::cout << std::format("\n\n*** STACK OVERFLOW ***\nTask: {} (handle 0x{:p})\n", task_name, reinterpret_cast<void*>(task));

    for (;;) { tight_loop_contents(); }
}

void vApplicationMallocFailedHook() {
    taskDISABLE_INTERRUPTS();

    std::cout << std::format("\n\n*** MALLOC FAILED ***\n");

    for (;;) { tight_loop_contents(); }
}
}
