cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_STDLIB_MODULES)
    return()
endif()

corekit_add_module(atomic
    INCLUDES     "inc"
    SOURCES      "src/atomic.cpp"
)

corekit_add_module(conditionvariable
    INCLUDES     "inc"
)

corekit_add_module(mutex
    INCLUDES     "inc"
)

corekit_add_module(semaphore
    INCLUDES     "inc"
    SOURCES      "src/semaphore.cpp"
    DEPENDENCIES "conditionvariable;mutex"
)