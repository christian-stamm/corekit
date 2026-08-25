cmake_minimum_required(VERSION 3.25)

corekit_add_module(atomic API
    INCLUDES   "inc"
    SOURCES    "src/atomic.cpp"
    TESTS      "test/atomic.cpp"
)

corekit_add_module(conditionvariable API
    INCLUDES   "inc"
    SOURCES    "src/conditionvariable.cpp"
)

corekit_add_module(mutex API
    INCLUDES   "inc"
    SOURCES    "src/mutex.cpp"
    TESTS      "test/mutex.cpp"
)

corekit_add_module(semaphore API
    INCLUDES   "inc"
    SOURCES    "src/semaphore.cpp"
    TESTS      "test/semaphore.cpp"
)