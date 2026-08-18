cmake_minimum_required(VERSION 3.25)

if(COREKIT_BASE_OS STREQUAL "linux")
  
    option(
        COREKIT_BUILD_PLATFORM_LINUX
        "Build Linux Platform"
        ON
    )

endif()