cmake_minimum_required(VERSION 3.25)

corekit_add_module(atomic API
    PUBLIC_INCLUDES  "inc"
    SOURCE_FILES     "src/atomic.cpp"
    GTEST_FILES      "test/atomic.cpp"
)

corekit_add_module(conditionvariable API
    PUBLIC_INCLUDES  "inc"
    SOURCE_FILES     "src/conditionvariable.cpp"
    GTEST_FILES      "test/conditionvariable.cpp"
)

corekit_add_module(mutex API
    PUBLIC_INCLUDES  "inc"
    SOURCE_FILES     "src/mutex.cpp"
    GTEST_FILES      "test/mutex.cpp"
)

corekit_add_module(semaphore API
    PUBLIC_INCLUDES  "inc"
    SOURCE_FILES     "src/semaphore.cpp"
    GTEST_FILES      "test/semaphore.cpp"
)