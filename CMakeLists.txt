cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(COREKIT_WORKDIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COREKIT_CMAKE_DIR "${COREKIT_WORKDIR}/cmake")
set(COREKIT_PAL_FILE "${COREKIT_CMAKE_DIR}/pal/${COREKIT_PAL}.cmake")
set(COREKIT_BSP_FILE "${COREKIT_CMAKE_DIR}/bsp/${COREKIT_BSP}.cmake")

include("${COREKIT_CMAKE_DIR}/core.cmake")

corekit_bsp_stage_0()
corekit_pal_stage_0()
corekit_bsp_stage_1()

project(corekit
    VERSION 0.0.1
    DESCRIPTION "COREKIT - A C++ Utility Library"
    LANGUAGES C CXX ASM
)

corekit_bsp_stage_2()
corekit_pal_stage_1()
corekit_bsp_stage_3()

add_subdirectory(corekit)
add_subdirectory(thirdparty)