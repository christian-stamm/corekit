cmake_minimum_required(VERSION 3.25)

corekit_add_impl(atomic
    INCLUDE_PATHS   inc
)

corekit_add_impl(conditionvariable
    INCLUDE_PATHS   inc
)


corekit_add_impl(mutex
    INCLUDE_PATHS   inc
)

corekit_add_impl(semaphore
    INCLUDE_PATHS   inc
    SOURCE_FILES    src/semaphore.cpp
    DEPENDENCIES    conditionvariable mutex
)