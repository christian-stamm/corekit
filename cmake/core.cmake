cmake_minimum_required(VERSION 3.25)

message(STATUS "COREKIT_PAL_FILE: ${COREKIT_PAL_FILE}")
message(STATUS "COREKIT_BSP_FILE: ${COREKIT_BSP_FILE}")
message(STATUS "COREKIT_COMMON_PATH: ${COREKIT_COMMON_PATH}")
message(STATUS "COREKIT_PLATFORM_PATH: ${COREKIT_PLATFORM_PATH}")
message(STATUS "COREKIT_BUILD_TEST: ${COREKIT_BUILD_TEST}")

if(NOT DEFINED COREKIT_PAL)
    message(FATAL_ERROR "COREKIT_PAL is not set. Choose from: stdlib, freertos, baremetal, zephyr, ...")
endif()

if(EXISTS ${COREKIT_PAL_FILE})
    include(${COREKIT_PAL_FILE})

    if( NOT COMMAND corekit_pal_stage_0 OR 
        NOT COMMAND corekit_pal_stage_1 )
        message(FATAL_ERROR "corekit_pal_stage_<0-3> function are not defined in ${COREKIT_PAL_FILE}. Please ensure that the PAL file defines them.")
    endif()

else()
    message(FATAL_ERROR "Unsupported PAL: ${COREKIT_PAL}. File not found: ${COREKIT_PAL_FILE}")
endif()
    
if(EXISTS ${COREKIT_BSP_FILE})
    include(${COREKIT_BSP_FILE})

    if( NOT COMMAND corekit_bsp_stage_0 OR 
        NOT COMMAND corekit_bsp_stage_1 OR
        NOT COMMAND corekit_bsp_stage_2 OR
        NOT COMMAND corekit_bsp_stage_3 )
        message(FATAL_ERROR "corekit_bsp_stage_<0-3> function are not defined in ${COREKIT_BSP_FILE}. Please ensure that the BSP file defines them.")
    endif()

else()
    message(FATAL_ERROR "Unsupported BSP: ${COREKIT_BSP}. File not found: ${COREKIT_BSP_FILE}")
endif()

if(NOT EXISTS ${COREKIT_PLATFORM_PATH})
    message(FATAL_ERROR "Specified platform '${COREKIT_PAL}' is not supported. Please provide platform implementation at ${COREKIT_PLATFORM_PATH}.")
endif()

if(NOT DEFINED COREKIT_BUILD_TEST)
    message(WARNING "COREKIT_BUILD_TEST is not set. Disabling tests.")
    option(COREKIT_BUILD_TEST "Build Tests" OFF)
endif()

if (COREKIT_BUILD_TEST)
    enable_testing()
endif()

# function(corekit_declare_capability capability)

#     add_library(corekit-${capability} INTERFACE)
#     add_library(corekit::${capability} ALIAS corekit-${capability})

#     string(TOUPPER "${capability}" CAPABILITY_UPPER)
#     option("COREKIT_HAS_${CAPABILITY_UPPER}" ON)
#     message(STATUS "Corekit capability corekit::${capability} available on platform '${COREKIT_PAL}'")
    
# endfunction()

# function(corekit_bind_capability capability)

#     if(NOT TARGET corekit::${capability})
#         message(FATAL_ERROR "Cannot bind capability. Capability '${capability}' is not declared.")
#     endif()

#     if(NOT TARGET corekit::impl::${capability})
#         message(FATAL_ERROR "Cannot bind capability. Implementation for '${capability}' does not exist.")
#     endif()

#     target_link_libraries(corekit::${capability} INTERFACE corekit::impl::${capability})
    
# endfunction()

# function(corekit_register_capability capability)

#     add_library(corekit-${capability} INTERFACE)
#     add_library(corekit::${capability} ALIAS corekit-${capability})

#     string(TOUPPER "${capability}" CAPABILITY_UPPER)
#     option("COREKIT_HAS_${CAPABILITY_UPPER}" ON)
#     message(STATUS "Corekit capability corekit::${capability} available on platform '${COREKIT_PAL}'")
    
# endfunction()

# include(${COREKIT_COMMON_PATH}/cmake/capabilites.cmake)
# include(${COREKIT_PLATFORM_PATH}/cmake/capabilites.cmake)

# set(COREKIT_CAPABILITIES
#     ${COREKIT_COMMON_CAPABILITIES}
#     ${COREKIT_PLATFORM_CAPABILITIES}
# )

# foreach(capability IN LISTS COREKIT_CAPABILITIES)
#     corekit_declare_capability(${capability})
# endforeach()

# function(corekit_declare_api capability impl_dir)

#     set(lib_name corekit-${capability})
#     set(inc_path ${impl_dir}/inc)
#     set(lib_impl corekit::impl::${capability})

#     if(NOT TARGET ${lib_impl})

#         message(WARNING 
#             "Cannot declare API. "
#             "Capability '${capability}' is not implemented on '${COREKIT_PAL}' platform."
#         )

#         return()

#     endif()
    
#     add_library(${lib_name} INTERFACE)

#     target_include_directories(${lib_name}
#         INTERFACE
#             ${inc_path}
#     )

#     target_link_libraries(${lib_name}
#         INTERFACE
#             ${lib_impl}
#     )

#     add_library(corekit::${capability} ALIAS ${lib_name})

#     string(TOUPPER "${capability}" CAPABILITY_UPPER)

#     option("COREKIT_HAS_${CAPABILITY_UPPER}" ON)

#     message(STATUS "API capability corekit::${capability} available on platform '${COREKIT_PAL}'")

# endfunction()

# function(corekit_declare_impl capability impl_dir dep_libs)

#     set(lib_name corekit-impl-${capability})
#     set(lib_type STATIC)
#     set(lib_impl PUBLIC)
#     set(inc_path ${impl_dir}/inc)
#     set(src_file ${impl_dir}/src/${capability}.cpp)

#     if (NOT EXISTS ${src_file})
#         set(lib_type INTERFACE)
#         set(lib_impl INTERFACE)
#         set(src_file "")
#     endif()

#     add_library(${lib_name}
#         ${lib_type}
#         ${src_file}
#     )

#     target_include_directories(${lib_name}
#         ${lib_impl}
#         ${inc_path}
#     )

#     target_link_libraries(${lib_name}
#         ${lib_impl}
#         ${dep_libs}
#     )

#     add_library(corekit::impl::${capability} ALIAS ${lib_name})

#     message(STATUS "declareed implementation for capability corekit::impl::${capability}")

# endfunction()