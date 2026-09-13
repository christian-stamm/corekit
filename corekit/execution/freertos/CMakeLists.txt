cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_FREERTOS_MODULES)
    return()
endif()

corekit_add_module(stoptoken
    INCLUDES     "inc"
    SOURCES      "src/stoptoken.cpp"
    DEPENDENCIES "atomic"
)

corekit_add_module(executor
    INCLUDES     "inc"
    SOURCES      "src/executor.cpp"
    DEPENDENCIES "FreeRTOS-Kernel;atomic;task;semaphore;stoptoken"
)