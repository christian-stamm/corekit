cmake_minimum_required(VERSION 3.25)

corekit_add_module(error API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/error.cpp"
    GTEST_FILES         "test/error.cpp"
)

corekit_add_module(assert API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/assert.cpp"
    GTEST_FILES         "test/assert.cpp"
    DEPENDENCIES        "error"
)

corekit_add_module(logger API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/logger.cpp"
    DEPENDENCIES        "serialdevice;streambuffer;mutex"
)

corekit_add_module(watch API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/watch.cpp"
    DEPENDENCIES        "time"
)