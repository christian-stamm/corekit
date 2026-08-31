cmake_minimum_required(VERSION 3.25)

if(${BUILD_FREERTOS_MODULES})

    corekit_add_module(time IMPL
        PUBLIC_INCLUDES     "inc"
        SOURCE_FILES        "src/time.cpp"
        DEPENDENCIES        "FreeRTOS-Kernel"
    )

endif()