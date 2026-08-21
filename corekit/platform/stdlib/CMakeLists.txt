cmake_minimum_required(VERSION 3.25)

corekit_module(atomic)
corekit_module(mutex)

corekit_module(semaphore 
    SOURCES src/semaphore.cpp
)

corekit_module(stoptoken)

corekit_module(thread
    SOURCES src/thread.cpp
    DEPENDS atomic stoptoken task
)

corekit_module(time)
