cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_FREERTOS_MODULES)
    return()
endif()

corekit_add_module(queue
    INCLUDES     "inc"
    SOURCES      "src/queue.cpp"
    DEPENDENCIES "check;FreeRTOS-Kernel"
)

corekit_add_module(stack
    INCLUDES     "inc"
    SOURCES      "src/stack.cpp"
    DEPENDENCIES "queue"
)