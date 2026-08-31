cmake_minimum_required(VERSION 3.25)

if(${BUILD_POSIX_MODULES})

        add_library(FreeRTOS-Kernel STATIC
                ${CMAKE_CURRENT_LIST_DIR}/src/port.c
                ${CMAKE_CURRENT_LIST_DIR}/src/wait_for_event.c
        )

        target_include_directories(FreeRTOS-Kernel PUBLIC
                ${CMAKE_CURRENT_LIST_DIR}/inc
        )

        target_link_libraries(FreeRTOS-Kernel PUBLIC
                FreeRTOS-Kernel-Core
        )

        target_compile_definitions(FreeRTOS-Kernel PUBLIC
                LIB_FREERTOS_KERNEL=1
                FREE_RTOS_KERNEL_SMP=1
        )

endif()