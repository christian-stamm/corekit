cmake_minimum_required(VERSION 3.25)

corekit_add_module(error
    INCLUDES     "inc"
    SOURCES      "src/error.cpp"
    TESTS        "test/error.cpp"
)

corekit_add_module(check
    INCLUDES     "inc"
    SOURCES      "src/check.cpp"
    DEPENDENCIES "atomic;error"
)

corekit_add_module(logger
    INCLUDES     "inc"
    SOURCES      "src/logger.cpp"
    DEPENDENCIES "serialdevice;streambuffer;mutex"
)

corekit_add_module(watch
    INCLUDES     "inc"
    SOURCES      "src/watch.cpp"
    DEPENDENCIES "time"
)