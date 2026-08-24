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

    make_paths_abs(sources ${ARG_SOURCES})
    make_paths_abs(includes "${CMAKE_CURRENT_SOURCE_DIR}/inc")

    # Remember module name.
    set_property(
        GLOBAL APPEND
        PROPERTY COREKIT_MODULES
        "${NAME}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${NAME}_SOURCES"
        "${sources}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${NAME}_INCLUDES"
        "${includes}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${NAME}_DEPENDS"
        "${ARG_DEPENDS}"
    )

endfunction()


function(corekit_finalize)
    # ----------------------------------------------------------
    # Pass 2: Configure targets
    #
    # At this point ALL corekit::* targets exist, so dep
    # ordering is irrelevant.
    # ----------------------------------------------------------

    get_property(
        modules
        GLOBAL
        PROPERTY COREKIT_MODULES
    )

    foreach(module IN LISTS modules)

        get_property(
            sources
            GLOBAL
            PROPERTY "COREKIT_${module}_SOURCES"
        )

        get_property(
            includes
            GLOBAL
            PROPERTY "COREKIT_${module}_INCLUDES"
        )

        get_property(
            depends
            GLOBAL
            PROPERTY "COREKIT_${module}_DEPENDS"
        )

        
        set(missing_deps "")

        foreach(dep IN LISTS depends)
            if(NOT "${dep}" MATCHES "^corekit::")
                continue()
            endif()

            if(NOT dep IN_LIST modules)
                list(APPEND missing_deps "${dep}")
            endif()
        endforeach()

        if(missing_deps)
            list(JOIN missing_deps "\n - " missing_deps_pretty)

            message(
                WARNING
                "corekit module '${module}' is not available due to missing dependencies:\n"
                " - ${missing_deps_pretty}"
            )

            continue()

        endif()

        set(lib_name "corekit_${module}")
        set(lib_type INTERFACE)
        set(lib_link INTERFACE)

        if(sources)
            set(lib_type STATIC)
            set(lib_link PUBLIC)
        endif()


        add_library(${lib_name} ${lib_type} ${sources})

        target_include_directories(${lib_name} ${lib_link} ${includes})

        foreach(dep IN LISTS depends)

            if(dep IN_LIST modules)
                set(dep "corekit::${dep}")
            endif()

            target_link_libraries("corekit_${module}" ${lib_link} "${dep}")
            message(STATUS "Linked corekit module '${module}' to '${dep}'")

        endforeach()


        add_library("corekit::${module}" ALIAS "${lib_name}")

        string(TOUPPER "${module}" module_upper)
        add_compile_definitions("COREKIT_HAS_${module_upper}=1")

    endforeach()
endfunction()