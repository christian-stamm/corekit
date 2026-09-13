cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_STDLIB_MODULES)
    return()
endif()

corekit_add_module(queue
    INCLUDES     "inc"
    SOURCES      "src/queue.cpp"
    DEPENDENCIES "conditionvariable;mutex"
)

corekit_add_module(stack
    INCLUDES     "inc"
    SOURCES      "src/stack.cpp"
    DEPENDENCIES "queue"
)