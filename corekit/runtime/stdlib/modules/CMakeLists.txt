cmake_minimum_required(VERSION 3.25)

corekit_add_module(atomic)
corekit_add_module(mutex)

corekit_add_module(semaphore 
    SOURCES src/semaphore.cpp
)

corekit_add_module(stoptoken)

corekit_add_module(thread
    SOURCES src/thread.cpp
    DEPENDS atomic stoptoken task
)

corekit_add_module(time)
