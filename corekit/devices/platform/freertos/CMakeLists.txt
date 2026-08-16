cmake_minimum_required(VERSION 3.25)

if(COREKIT_PLATFORM STREQUAL "PICO")

    add_library(corekit-devices-pico
        ${CMAKE_CURRENT_SOURCE_DIR}/src/uart.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pio.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/spi.cpp
    )

    target_include_directories(corekit-devices-pico
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-devices-pico
        PUBLIC
            hardware_uart
            hardware_spi
            hardware_pio
            hardware_i2c
            hardware_gpio
            hardware_dma
            hardware_irq

            corekit::system
            corekit::devices::common
    )

    add_library(corekit::devices ALIAS corekit-devices-pico)  

    if(COREKIT_BUILD_PTEST STREQUAL "ON")

        add_executable(corekit-devices-pico-test
            ${CMAKE_CURRENT_SOURCE_DIR}/test/hello_lib.cpp
        )

        target_link_libraries(corekit-devices-pico-test
            PUBLIC
                corekit::devices
                corekit::monitor
                corekit::system
        )

    endif()

endif()