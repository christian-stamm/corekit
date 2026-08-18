cmake_minimum_required(VERSION 3.25)

corekit_add_module(stoptoken API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/stoptoken.cpp"
    GTEST_FILES         "test/stoptoken.cpp"
)

corekit_add_module(executor API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/executor.cpp"
    DEPENDENCIES        "task"
)