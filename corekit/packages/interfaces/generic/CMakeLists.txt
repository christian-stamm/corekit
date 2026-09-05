cmake_minimum_required(VERSION 3.25)

corekit_add_module(basedevice API IMPL
    PUBLIC_INCLUDES      "inc"
    SOURCE_FILES         "src/basedevice.cpp"
    DEPENDENCIES         "atomic;result;watch"
)

corekit_add_module(serialdevice API IMPL
    PUBLIC_INCLUDES      "inc"
    SOURCE_FILES         "src/serialdevice.cpp"
    DEPENDENCIES         "basedevice"
)

corekit_add_module(streambuffer API IMPL
    PUBLIC_INCLUDES      "inc"
    SOURCE_FILES         "src/streambuffer.cpp"
    DEPENDENCIES         "serialdevice"
)