cmake_minimum_required(VERSION 3.25)

macro(corekit_bsp_bootstrap)

    set(PICO_BOARD "bernd")
    
    set(rp2040_boards
        pico
        pico_w
    )

    set(rp2350_boards
        pico2
        pico2_w    
        bernd
    )

    set(rp2XXX_boards ${rp2040_boards} ${rp2350_boards})
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
    set(PICO_BOARD_HEADER_DIRS ${COREKIT_ROOT}/thirdparty/raspberry/pico/boards)

    if (PICO_BOARD IN_LIST rp2040_boards)
        set(PICO_PLATFORM "rp2040")
        set(RTOS_PORTABLE "rp2040")
    elseif (PICO_BOARD IN_LIST rp2350_boards)
        set(PICO_PLATFORM "rp2350-arm-s")
        set(RTOS_PORTABLE "rp2350/arm")
    else()
        message(FATAL_ERROR "Invalid target '${PICO_BOARD}' for the RP2XXX series. Supported targets: ${rp2XXX_boards}")
    endif()

    if(NOT EXISTS ${PICO_SDK_PATH})
        message(FATAL_ERROR "PICO_SDK_PATH='${PICO_SDK_PATH}' points to a non-existent directory. Please set it to the path of the Pico SDK.")
    endif()

    set(PICO_SDK_IMPORT_FILE "${PICO_SDK_PATH}/external/pico_sdk_import.cmake")

    if(NOT EXISTS ${PICO_SDK_IMPORT_FILE})
        message(FATAL_ERROR "'${PICO_SDK_IMPORT_FILE}' does not contain the Pico SDK import config. Please set it to the path of the Pico SDK.")
    endif()


    set(PICO_RTOS_PORTABLE_PATH "${COREKIT_ROOT}/thirdparty/freertos/pico/${RTOS_PORTABLE}")

    if(NOT EXISTS ${PICO_RTOS_PORTABLE_PATH})
        message(FATAL_ERROR "PICO_RTOS_PORTABLE_PATH='${PICO_RTOS_PORTABLE_PATH}' points to a non-existent directory. Please set it to the path of the FreeRTOS portable layer for the target platform.")
    endif()

    include(${PICO_SDK_IMPORT_FILE})


endmacro()

macro(corekit_bsp_launch)

    set(BUILD_PICO_MODULES TRUE)
    pico_sdk_init()

    add_subdirectory(${PICO_RTOS_PORTABLE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/freertos)

endmacro()