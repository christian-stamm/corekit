cmake_minimum_required(VERSION 3.25)

corekit_add_module(stoptoken API
    INCLUDES   "inc"
    SOURCES    "src/stoptoken.cpp"
    TESTS      "test/stoptoken.cpp"
)

corekit_add_module(task API IMPL
    INCLUDES   "inc"
    SOURCES    "src/task.cpp"
    TESTS      "test/task.cpp"
    DEPENDS    atomic core stoptoken
)

corekit_add_module(thread API
    INCLUDES   "inc"
    SOURCES    "src/thread.cpp"
    TESTS      "test/thread.cpp"
)