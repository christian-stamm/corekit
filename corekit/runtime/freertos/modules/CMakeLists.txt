cmake_minimum_required(VERSION 3.25)

corekit_add_module(atomic
    DEPENDS FreeRTOS-Kernel-Heap4
)

corekit_add_module(mutex
    SOURCES src/mutex.cpp
    DEPENDS FreeRTOS-Kernel-Heap4 semaphore
)

corekit_add_module(semaphore 
    SOURCES src/semaphore.cpp
    DEPENDS FreeRTOS-Kernel-Heap4
)

corekit_add_module(stoptoken
    SOURCES src/stoptoken.cpp
    DEPENDS FreeRTOS-Kernel-Heap4 mutex atomic
)

corekit_add_module(time
    SOURCES src/time.cpp
    DEPENDS FreeRTOS-Kernel-Heap4
)
