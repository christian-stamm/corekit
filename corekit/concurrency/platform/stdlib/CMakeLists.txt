cmake_minimum_required(VERSION 3.25)

corekit_add_impl(atomic)
corekit_add_impl(conditionvariable)
corekit_add_impl(mutex)
corekit_add_impl(semaphore
    SOURCES src/semaphore.cpp
    DEPENDS conditionvariable mutex
)