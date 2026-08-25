cmake_minimum_required(VERSION 3.25)

corekit_add_module(atomic
    INCLUDE_PATHS "inc"
    TEST_FILES test/atomic.cpp
)

corekit_add_module(conditionvariable
    INCLUDE_PATHS "inc"
)

corekit_add_module(mutex
    INCLUDE_PATHS "inc"
    TEST_FILES test/mutex.cpp
)

corekit_add_module(semaphore
    INCLUDE_PATHS "inc"
    TEST_FILES test/semaphore.cpp
)