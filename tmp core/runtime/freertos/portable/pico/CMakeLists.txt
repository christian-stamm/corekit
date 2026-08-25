cmake_minimum_required(VERSION 3.25)

set(rp2040_boards
    pico
    pico_w
)

set(rp2350_boards
    pico2
    pico2_w
    bernd
)

set(PICO_BOARD ${COREKIT_BOARD})
set(PICO_CXX_ENABLE_EXCEPTIONS ON)
set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
set(PICO_BOARD_HEADER_DIRS ${COREKIT_WORKDIR}/thirdparty/raspberry/pico/boards)
set(PICO_SDK_IMPORT_FILE "${PICO_SDK_PATH}/external/pico_sdk_import.cmake" CACHE INTERNAL "")

if (NOT EXISTS $ENV{PICO_SDK_PATH})
    message(FATAL_ERROR "No valid PICO_SDK_PATH is set. Scanned path '$ENV{PICO_SDK_PATH}'.")
endif()

if(NOT EXISTS ${PICO_SDK_IMPORT_FILE})
    message(FATAL_ERROR "PICO_SDK_IMPORT_FILE does not exist: ${PICO_SDK_IMPORT_FILE}. Please ensure that PICO_SDK_PATH is set correctly.")
endif()

if (${COREKIT_BOARD} IN_LIST rp2040_boards)
    set(PICO_PLATFORM "rp2040")
elseif (${COREKIT_BOARD} IN_LIST rp2350_boards)
    set(PICO_PLATFORM "rp2350-arm-s")
else()
    message(FATAL_ERROR "Unsupported PICO_BOARD: ${COREKIT_BOARD}. Supported boards are: ${rp2040_boards}; ${rp2350_boards}.")
endif()
    
include(${PICO_SDK_IMPORT_FILE})
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${PICO_PLATFORM}")

pico_sdk_init()