cmake_minimum_required(VERSION 3.25)

corekit_add_module(core)

corekit_add_module(task 
    SOURCES src/task.cpp 
    DEPENDS atomic stoptoken
)

corekit_add_module(watch 
    SOURCES src/watch.cpp 
    DEPENDS time
)

corekit_add_module(basedevice
    SOURCES src/basedevice.cpp 
    DEPENDS atomic watch 
)

corekit_add_module(serialdevice
    SOURCES src/serialdevice.cpp 
    DEPENDS basedevice
)