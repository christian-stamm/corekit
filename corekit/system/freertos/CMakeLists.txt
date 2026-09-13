cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_FREERTOS_MODULES)
    return()
endif()

corekit_add_module(time
    INCLUDES     "inc"
    SOURCES      "src/time.cpp"
    DEPENDENCIES "FreeRTOS-Kernel"
)

corekit_add_module(memory
    INCLUDES     "inc"
    SOURCES      "src/memory.cpp"
    DEPENDENCIES "FreeRTOS-Kernel;check"
)