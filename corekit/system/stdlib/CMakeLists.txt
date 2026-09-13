cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_STDLIB_MODULES)
    return()
endif()

corekit_add_module(time
    INCLUDES "inc"
    SOURCES  "src/time.cpp"
)

corekit_add_module(memory
    INCLUDES     "inc"
    SOURCES      "src/memory.cpp"
    DEPENDENCIES "check"
)