cmake_minimum_required(VERSION 3.25)

if(COREKIT_BASE_OS STREQUAL "baremetal")

    option(
        COREKIT_BUILD_PLATFORM_BAREMETAL 
        "Build for Baremetal" 
        ON
    )

    include("${COREKIT_CMAKE_DIR}/${COREKIT_CMAKE_SUBDIR}/bootstrap.cmake")
      
endif()