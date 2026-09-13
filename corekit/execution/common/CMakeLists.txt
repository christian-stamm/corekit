cmake_minimum_required(VERSION 3.25)

corekit_add_module(task
    INCLUDES     "inc"
    SOURCES      "src/task.cpp"
    TESTS        "test/task.cpp"
    DEPENDENCIES "atomic;check;logger;stoptoken;"
)