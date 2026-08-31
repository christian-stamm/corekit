cmake_minimum_required(VERSION 3.25)

macro(corekit_bsp_bootstrap)

    set(PICO_BOARD ${COREKIT_TARGET_DEVICE})

    set(BUILD_RP2XXX_MODULES OFF)
    set(BUILD_RP2040_MODULES OFF)
    set(BUILD_RP2350_MODULES OFF)
    
    set(rp2040_boards
        pico
        pico_w
    )

    set(rp2350_boards
        pico2
        pico2_w    
        bernd
    )

    if (PICO_BOARD IN_LIST rp2040_boards)
        set(BUILD_RP2040_MODULES ON)
        set(PICO_PLATFORM "rp2040")
        set(RTOS_PORTABLE "pico/rp2040")
    elseif (PICO_BOARD IN_LIST rp2350_boards)
        set(BUILD_RP2350_MODULES ON)
        set(PICO_PLATFORM "rp2350-arm-s")
        set(RTOS_PORTABLE "pico/rp2350/arm")
    endif()

    if(NOT BUILD_RP2040_MODULES AND NOT BUILD_RP2350_MODULES)
        message(FATAL_ERROR "Invalid target '${PICO_BOARD}' for the RP2XXX series. Supported targets: ${rp2XXX_boards}")
    endif()

    set(BUILD_RP2XXX_MODULES ON)
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
    set(PICO_BOARD_HEADER_DIRS ${COREKIT_ROOT}/thirdparty/raspberry/pico/boards)
    set(PICO_SDK_IMPORT_FILE "${PICO_SDK_PATH}/external/pico_sdk_import.cmake")

    if(NOT EXISTS ${PICO_SDK_IMPORT_FILE})
        message(FATAL_ERROR "'${PICO_SDK_IMPORT_FILE}' does not contain the Pico SDK import config. Please set it to the path of the Pico SDK.")
    endif()

    include(${PICO_SDK_IMPORT_FILE})

endmacro()

macro(corekit_bsp_launch)

    pico_sdk_init()

endmacro()