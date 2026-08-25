cmake_minimum_required(VERSION 3.25)

corekit_add_mod_impl(atomic)
corekit_add_mod_impl(conditionvariable)
corekit_add_mod_impl(mutex)
corekit_add_mod_impl(semaphore
    SOURCES src/semaphore.cpp
    DEPENDS conditionvariable mutex
)