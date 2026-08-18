cmake_minimum_required(VERSION 3.25)

if(COREKIT_BUILD_PLATFORM_FREERTOS)

    add_library(corekit-peripherals-platform-freertos INTERFACE)

    target_include_directories(corekit-peripherals-platform-freertos
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-peripherals-platform-freertos
        INTERFACE
            corekit::system
            corekit::peripherals::common
    )

    add_library(corekit::peripherals ALIAS corekit-peripherals-platform-freertos)  

    if(COREKIT_BUILD_TEST)

        add_executable(corekit-peripherals-platform-freertos-test
            ${CMAKE_CURRENT_SOURCE_DIR}/test/hello_lib.cpp
        )

        target_link_libraries(corekit-peripherals-platform-freertos-test
            PUBLIC
                corekit::peripherals
                corekit::logging
                corekit::system
        )

    endif()

endif()