cmake_minimum_required(VERSION 3.25)

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