# Copyright (c) 2020 Raspberry Pi (Trading) Ltd.
#
# SPDX-License-Identifier: BSD-3-Clause

# Called after the Raspberry Pi Pico SDK has been initialized to add our libraries

if(${BUILD_RP2350_MODULES})

        if (PICO_SDK_VERSION_STRING VERSION_GREATER_EQUAL "1.3.2")
        target_compile_definitions(FreeRTOS-Kernel-Core INTERFACE
                PICO_CONFIG_RTOS_ADAPTER_HEADER=${CMAKE_CURRENT_LIST_DIR}/inc/freertos_sdk_config.h)
        endif()

        add_library(FreeRTOS-Kernel STATIC
                ${CMAKE_CURRENT_LIST_DIR}/src/port.c
                ${CMAKE_CURRENT_LIST_DIR}/src/portasm.c
        )

        target_include_directories(FreeRTOS-Kernel PUBLIC
                ${CMAKE_CURRENT_LIST_DIR}/inc
        )

        target_link_libraries(FreeRTOS-Kernel PUBLIC
                FreeRTOS-Kernel-Core
                pico_base_headers
                pico_multicore
                pico_time
                hardware_clocks
                hardware_exception
        )

        target_compile_definitions(FreeRTOS-Kernel PUBLIC
                LIB_FREERTOS_KERNEL=1
                FREE_RTOS_KERNEL_SMP=1
        )

endif()