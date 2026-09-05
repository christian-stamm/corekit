cmake_minimum_required(VERSION 3.25)

corekit_add_module(stoptoken IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/stoptoken.cpp"
    DEPENDENCIES        "atomic"
)

corekit_add_module(executor IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/executor.cpp"
    DEPENDENCIES        "FreeRTOS-Kernel-Heap4;stoptoken;semaphore;task;time"
)