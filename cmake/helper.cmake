cmake_minimum_required(VERSION 3.25)

function(make_paths_abs ABS_PATHS)
    set(paths "")

    foreach(path IN LISTS ARGN)
        if(IS_ABSOLUTE "${path}")
            list(APPEND paths "${path}")
        else()
            list(APPEND paths "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
        endif()
    endforeach()

    set(${ABS_PATHS} "${paths}" PARENT_SCOPE)
endfunction()
