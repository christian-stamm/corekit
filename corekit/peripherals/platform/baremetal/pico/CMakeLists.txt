cmake_minimum_required(VERSION 3.25)

if(COREKIT_BUILD_PLATFORM_FREERTOS)

    add_library(corekit-peripherals-platform-freertos
        ${CMAKE_CURRENT_SOURCE_DIR}/src/uart.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pio.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/spi.cpp
    )

    target_include_directories(corekit-peripherals-platform-freertos
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-peripherals-platform-freertos
        PUBLIC
            hardware_uart
            hardware_spi
            hardware_pio
            hardware_i2c
            hardware_gpio
            hardware_dma
            hardware_irq

            corekit::system
            corekit::peripherals::common
    )

    add_library(corekit::peripherals ALIAS corekit-peripherals-platform-freertos)  

    if(COREKIT_BUILD_PTEST STREQUAL "ON")

        add_executable(corekit-peripherals-platform-freertos-test
            ${CMAKE_CURRENT_SOURCE_DIR}/test/hello_lib.cpp
        )

        target_link_libraries(corekit-peripherals-platform-freertos-test
            PUBLIC
                corekit::devices
                corekit::monitor
                corekit::system
        )

    endif()

endif()