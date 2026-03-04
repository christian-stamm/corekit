cmake_minimum_required(VERSION 3.25)

########################################################################

if(BUILD_COREKIT_OPENGL)

    include(FindPkgConfig)
    find_package(glm REQUIRED)
    find_package(OpenCV REQUIRED)
    find_package(Freetype REQUIRED)
   
    add_library(corekit-opengl STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/assert.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/program.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/texture.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/window.cpp
    )

    target_include_directories(corekit-opengl
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
            ${GLM_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
            ${FREETYPE_INCLUDE_DIRS}
    )

    target_link_libraries(corekit-opengl
        PUBLIC
            glad
            ${OpenCV_LIBS}
            ${GLM_LIBRARIES}
            ${FREETYPE_LIBRARIES}
            corekit-core
    )

    if(TARGET trinity::platform)
        target_link_libraries(corekit-opengl PUBLIC trinity::platform)
    endif()

    add_library(corekit::opengl ALIAS corekit-opengl)

endif()