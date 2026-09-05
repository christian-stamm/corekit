cmake_minimum_required(VERSION 3.25)

corekit_add_module(queue IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/queue.cpp"
    DEPENDENCIES        "FreeRTOS-Kernel-Heap4"
)

corekit_add_module(stack IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/stack.cpp"
    DEPENDENCIES        "queue"
)