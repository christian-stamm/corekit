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


function(resolve_dependencies required_deps available_libs matched_out missing_out)
    set(matched "")
    set(missing "")

    foreach(dep IN LISTS ${required_deps})
        if(dep IN_LIST ${available_libs})
            list(APPEND matched "corekit::${dep}")
        elseif(TARGET "${dep}")
            list(APPEND matched "${dep}")
        else()
            list(APPEND missing "${dep}")
        endif()
    endforeach()

    set(${matched_out} "${matched}" PARENT_SCOPE)
    set(${missing_out} "${missing}" PARENT_SCOPE)
endfunction()
