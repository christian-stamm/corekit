cmake_minimum_required(VERSION 3.25)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/runtime)

if(TARGET corekit_bootstrap)
    corekit_runtime_bootstrap()
endif()

project(corekit-${COREKIT_RUNTIME}-${COREKIT_TARGET}
    VERSION 0.0.1
    DESCRIPTION "COREKIT - A C++ Utility Library"
    LANGUAGES C CXX ASM
)

if(TARGET corekit_bootstrap)
    corekit_runtime_configure()
endif()

corekit_finalize()
