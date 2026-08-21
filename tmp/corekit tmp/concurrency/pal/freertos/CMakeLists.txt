cmake_minimum_required(VERSION 3.25)

if(COREKIT_PAL_FREERTOS)

    add_library(corekit-concurrency-pal-freertos 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/stptkn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/time.cpp
    )

    target_include_directories(corekit-concurrency-pal-freertos 
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-concurrency-pal-freertos 
        PUBLIC
            corekit::concurrency::api
            FreeRTOS-Kernel-Heap4
    )

    add_library(corekit::concurrency::pal ALIAS corekit-concurrency-pal-freertos)

    # add_executable(corekit-concurrency-pal-freertos-test
    #     ${CMAKE_CURRENT_SOURCE_DIR}/test/basetest.cpp
    # )

    # target_link_libraries(corekit-concurrency-pal-freertos-test PUBLIC 
    #     corekit::concurrency
    # )

endif()