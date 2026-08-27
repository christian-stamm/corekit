cmake_minimum_required(VERSION 3.25)

corekit_add_module(atomic IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/atomic.cpp"
    DEPENDENCIES        "FreeRTOS-Kernel-Heap4"
)

corekit_add_module(conditionvariable IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/conditionvariable.cpp"
    DEPENDENCIES        "FreeRTOS-Kernel-Heap4;mutex"
)

corekit_add_module(mutex IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/mutex.cpp"
    DEPENDENCIES        "FreeRTOS-Kernel-Heap4;semaphore"
)

corekit_add_module(semaphore IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/semaphore.cpp"
    DEPENDENCIES        "FreeRTOS-Kernel-Heap4"
)