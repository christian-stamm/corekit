cmake_minimum_required(VERSION 3.25)

corekit_add_impl(task
    INCLUDE_PATHS   inc
    SOURCE_FILES    src/task.cpp
    DEPENDENCIES    atomic stoptoken
)