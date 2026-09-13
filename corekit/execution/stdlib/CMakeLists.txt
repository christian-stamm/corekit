cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_STDLIB_MODULES)
    return()
endif()

corekit_add_module(stoptoken
    INCLUDES     "inc"
)

corekit_add_module(executor
    INCLUDES     "inc"
    SOURCES      "src/executor.cpp"
    DEPENDENCIES "queue;semaphore;result;stoptoken"
)