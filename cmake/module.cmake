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

function(corekit_module NAME)
    # ----------------------------------------------------------
    # Pass 1: Create ALL targets
    # ----------------------------------------------------------

    cmake_parse_arguments(
        ARG                         # prefix
        ""                          # options
        ""                          # one-value arguments
        "SOURCES;DEPENDS"           # multi-value arguments
        ${ARGN}                     # arguments to parse
    )

    # Remember module name.
    set_property(
        GLOBAL APPEND
        PROPERTY COREKIT_MODULES
        "${NAME}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${NAME}_DEPENDS"
        "${ARG_DEPENDS}"
    )

    make_paths_abs(sources ${ARG_SOURCES})

    set(lib_name "corekit_${NAME}")
    set(lib_type INTERFACE)

    if(sources)
        set(lib_type STATIC)
    endif()

    add_library(${lib_name} ${lib_type} ${sources})

    if(lib_type STREQUAL "STATIC")
        set(lib_type PUBLIC)
    endif()

    target_include_directories(${lib_name} ${lib_type} ${CMAKE_CURRENT_SOURCE_DIR}/inc)

    add_library("corekit::${NAME}" ALIAS "${lib_name}")

endfunction()


function(corekit_finalize)
    # ----------------------------------------------------------
    # Pass 2: Configure targets
    #
    # At this point ALL corekit::* targets exist, so dependency
    # ordering is irrelevant.
    # ----------------------------------------------------------

    get_property(
        modules
        GLOBAL
        PROPERTY COREKIT_MODULES
    )

    foreach(module IN LISTS modules)

        get_property(
            depends
            GLOBAL
            PROPERTY "COREKIT_${module}_DEPENDS"
        )

        set(missing_depends "")

        foreach(dependency IN LISTS depends)

            if(NOT TARGET ${dependency} AND NOT TARGET "corekit::${dependency}")
                list(APPEND missing_depends "${dependency}")
                continue()
            endif()

            target_link_libraries(
                "corekit_${module}"
                PUBLIC
                    "corekit::${dependency}"
            )

        endforeach()

        list(JOIN missing_depends "\n - " missing_depends_pretty)

        if(missing_depends)
            message(
                WARNING
                "corekit module '${module}' is not available due to missing dependencies:\n"
                " - ${missing_depends_pretty}"
            )
        else()
            string(TOUPPER "${module}" module)
            option(
                "COREKIT_HAS_${module}"
                "COREKIT module '${module}' is available."
                ON
            )
        endif()

    endforeach()
endfunction()