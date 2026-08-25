cmake_minimum_required(VERSION 3.25)

corekit_add_module(atomic
    INCLUDES   "inc"
    TESTS      "test/atomic.cpp"
)

corekit_add_module(conditionvariable
    INCLUDES   "inc"
)

corekit_add_module(mutex
    INCLUDES   "inc"
    TESTS      "test/mutex.cpp"
)

corekit_add_module(semaphore
    INCLUDES   "inc"
    TESTS      "test/semaphore.cpp"
)