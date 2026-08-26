cmake_minimum_required(VERSION 3.25)

corekit_add_module(task API IMPL
    INCLUDES   "inc"
    SOURCES    "src/task.cpp"
    TESTS      "test/task.cpp"
    DEPENDS    atomic core stoptoken
)