cmake_minimum_required(VERSION 3.25)

corekit_add_module(basedevice
    INCLUDES      "inc"
    SOURCES       "src/basedevice.cpp"
    DEPENDENCIES  "atomic;check;watch"
)

corekit_add_module(serialdevice
    INCLUDES      "inc"
    SOURCES       "src/serialdevice.cpp"
    DEPENDENCIES  "basedevice"
)

corekit_add_module(streambuffer
    INCLUDES      "inc"
    SOURCES       "src/streambuffer.cpp"
    DEPENDENCIES  "serialdevice"
)