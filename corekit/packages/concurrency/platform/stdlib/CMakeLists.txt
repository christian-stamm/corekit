cmake_minimum_required(VERSION 3.25)

corekit_add_module(atomic IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/atomic.cpp"
)

corekit_add_module(conditionvariable IMPL
    PUBLIC_INCLUDES     "inc"
)

corekit_add_module(mutex IMPL
    PUBLIC_INCLUDES     "inc"
)

corekit_add_module(semaphore IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/semaphore.cpp"
    DEPENDENCIES        conditionvariable mutex
)