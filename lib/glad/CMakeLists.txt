cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(glad 
    VERSION 0.0.1
    LANGUAGES C
    DESCRIPTION "GLAD"
)

include(FindPkgConfig)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

add_library(${PROJECT_NAME} STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/glad.c"
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    glfw
    ${OpenGL_LIBRARIES}
)