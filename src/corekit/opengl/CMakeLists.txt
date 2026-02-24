cmake_minimum_required(VERSION 3.25)

########################################################################

if(BUILD_COREKIT_OPENGL)

    include(FindPkgConfig)
    find_package(glm REQUIRED)
    find_package(OpenCV REQUIRED)
   
    add_library(corekit-opengl STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/assert.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/program.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/texture.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/window.cpp
    )

    add_dependencies(
        corekit-core
        glad
    )
    
    target_include_directories(corekit-opengl PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${GLM_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )


    target_link_libraries(corekit-opengl PUBLIC
        glad
        ${OpenCV_LIBS}
        ${GLM_LIBRARIES}
        corekit-core
    )

endif()