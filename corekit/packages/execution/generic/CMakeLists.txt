cmake_minimum_required(VERSION 3.25)

corekit_add_module(task API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/task.cpp"
    GTEST_FILES         "test/task.cpp"
    DEPENDENCIES        "atomic;logger;result;stoptoken;"
)

corekit_add_module(monitor API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/monitor.cpp"
    GTEST_FILES         "test/monitor.cpp"
    DEPENDENCIES        "error;logger;task;time;"
)