cmake_minimum_required(VERSION 3.25)
set(MODULE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

corekit_add_module(atomic
    INCLUDES        "inc"
)

corekit_add_module(conditionvariable
    INCLUDES        "inc"
)

corekit_add_module(mutex
    INCLUDES        "inc"
)

corekit_add_module(semaphore
    INCLUDES        "inc"
    SOURCES         "src/semaphore.cpp"
    DEPENDS         conditionvariable mutex
)