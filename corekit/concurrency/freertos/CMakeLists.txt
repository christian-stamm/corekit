cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_FREERTOS_MODULES)
    return()
endif()

corekit_add_module(atomic
    INCLUDES     "inc"
    SOURCES      "src/atomic.cpp"
    DEPENDENCIES "FreeRTOS-Kernel"
)

corekit_add_module(conditionvariable
    INCLUDES     "inc"
    SOURCES      "src/conditionvariable.cpp"
    DEPENDENCIES "check;mutex;semaphore;queue"
)

corekit_add_module(mutex
    INCLUDES     "inc"
    SOURCES      "src/mutex.cpp"
    DEPENDENCIES "check;FreeRTOS-Kernel"
)

corekit_add_module(semaphore
    INCLUDES     "inc"
    SOURCES      "src/semaphore.cpp"
    DEPENDENCIES "check;FreeRTOS-Kernel"
)