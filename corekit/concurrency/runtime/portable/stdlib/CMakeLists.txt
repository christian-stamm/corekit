cmake_minimum_required(VERSION 3.25)
set(MODULE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

corekit_add_module(atomic IMPL
    INCLUDES        "inc"
)

corekit_add_module(conditionvariable IMPL
    INCLUDES        "inc"
)

corekit_add_module(mutex IMPL
    INCLUDES        "inc"
)

corekit_add_module(semaphore IMPL
    INCLUDES        "inc"
    SOURCES         "src/semaphore.cpp"
    DEPENDS         conditionvariable mutex
)