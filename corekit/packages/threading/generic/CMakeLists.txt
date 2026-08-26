cmake_minimum_required(VERSION 3.25)

corekit_add_module(task API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/task.cpp"
    GTEST_FILES         "test/task.cpp"
    DEPENDENCIES        "assert;atomic;stoptoken"
)