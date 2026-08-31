# Copyright (c) 2020 Raspberry Pi (Trading) Ltd.
#
# SPDX-License-Identifier: BSD-3-Clause

# Called after the Raspberry Pi Pico SDK has been initialized to add our libraries

set(FREERTOS_KERNEL_HEAP 4)
set(FREERTOS_KERNEL_PATH $ENV{FREERTOS_KERNEL_PATH})
set(FREERTOS_CONFIG_FILE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/inc)

add_library(FreeRTOS-Kernel-Core INTERFACE)

target_include_directories(FreeRTOS-Kernel-Core INTERFACE 
        ${FREERTOS_KERNEL_PATH}/include
        ${FREERTOS_CONFIG_FILE_DIRECTORY}
)

target_sources(FreeRTOS-Kernel-Core INTERFACE
        # CORE KERNEL
        ${FREERTOS_KERNEL_PATH}/croutine.c
        ${FREERTOS_KERNEL_PATH}/event_groups.c
        ${FREERTOS_KERNEL_PATH}/list.c
        ${FREERTOS_KERNEL_PATH}/queue.c
        ${FREERTOS_KERNEL_PATH}/stream_buffer.c
        ${FREERTOS_KERNEL_PATH}/tasks.c
        ${FREERTOS_KERNEL_PATH}/timers.c
        # MEMORY MANAGEMENT
        ${FREERTOS_KERNEL_PATH}/portable/MemMang/heap_${FREERTOS_KERNEL_HEAP}.c
        ${CMAKE_CURRENT_LIST_DIR}/src/rtoshook.cpp
)