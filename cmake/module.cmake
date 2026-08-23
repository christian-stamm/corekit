cmake_minimum_required(VERSION 3.25)

function(corekit_add_module NAME)
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

            if(TARGET ${dependency})
                target_link_libraries(
                    "corekit_${module}"
                    PUBLIC
                        "corekit::${dependency}"
                )

            elseif(TARGET "corekit::${dependency}")
                target_link_libraries(
                    "corekit_${module}"
                    PUBLIC
                        "corekit::${dependency}"
                )

            else()
                list(APPEND missing_depends "${dependency}")

            endif()

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