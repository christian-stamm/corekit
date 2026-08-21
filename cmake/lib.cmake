function(corekit_component NAME)
    cmake_parse_arguments(
        ARG
        ""
        ""
        "SOURCES;DEPENDS"
        ${ARGN}
    )

    # Remember component name.
    set_property(
        GLOBAL APPEND
        PROPERTY COREKIT_COMPONENTS
        "${NAME}"
    )

    # Store absolute source paths because finalize() may execute
    # from a completely different directory.
    set(absolute_sources "")

    foreach(source IN LISTS ARG_SOURCES)
        if(IS_ABSOLUTE "${source}")
            list(APPEND absolute_sources "${source}")
        else()
            list(APPEND absolute_sources
                "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            )
        endif()
    endforeach()

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${NAME}_SOURCES"
        "${absolute_sources}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${NAME}_DEPENDS"
        "${ARG_DEPENDS}"
    )
endfunction()


function(corekit_finalize)
    get_property(
        components
        GLOBAL
        PROPERTY COREKIT_COMPONENTS
    )

    # ----------------------------------------------------------
    # Pass 1: Create ALL targets
    # ----------------------------------------------------------

    foreach(component IN LISTS components)

        add_library(
            "corekit_${component}"
            STATIC
        )

        add_library(
            "corekit::${component}"
            ALIAS "corekit_${component}"
        )

    endforeach()


    # ----------------------------------------------------------
    # Pass 2: Configure targets
    #
    # At this point ALL corekit::* targets exist, so dependency
    # ordering is irrelevant.
    # ----------------------------------------------------------

    foreach(component IN LISTS components)

        get_property(
            sources
            GLOBAL
            PROPERTY "COREKIT_${component}_SOURCES"
        )

        get_property(
            depends
            GLOBAL
            PROPERTY "COREKIT_${component}_DEPENDS"
        )

        if(sources)
            target_sources(
                "corekit_${component}"
                PRIVATE
                    ${sources}
            )
        endif()

        foreach(dependency IN LISTS depends)

            if(NOT TARGET "corekit::${dependency}")
                message(
                    FATAL_ERROR
                    "corekit component '${component}' requires "
                    "'${dependency}', but that capability is not available."
                )
            endif()

            target_link_libraries(
                "corekit_${component}"
                PUBLIC
                    "corekit::${dependency}"
            )

        endforeach()

    endforeach()
endfunction()