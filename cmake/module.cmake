cmake_minimum_required(VERSION 3.25)

function(corekit_add_module NAME)
    # ----------------------------------------------------------
    # Pass 1: Create ALL targets
    # ----------------------------------------------------------

    cmake_parse_arguments(
        ARG                                         # prefix
        ""                                          # options
        ""                                          # one-value arguments
        "INCLUDE_PATHS;DEPENDENCIES;TEST_FILES"     # multi-value arguments
        ${ARGN}                                     # arguments to parse
    )

    # Remember module name.
    set_property(
        GLOBAL APPEND
        PROPERTY COREKIT_API_MODULES
        "${NAME}"
    )

    make_paths_abs(includes ${ARG_INCLUDE_PATHS})
    make_paths_abs(tests ${ARG_TEST_FILES})

    set_property(
        GLOBAL APPEND
        PROPERTY "COREKIT_API_${NAME}_INCLUDE_PATHS"
        "${includes}"
    )

    set_property(
        GLOBAL APPEND
        PROPERTY "COREKIT_API_${NAME}_DEPENDENCIES"
        "${ARG_DEPENDENCIES}"
    )

    set_property(
        GLOBAL APPEND
        PROPERTY "COREKIT_API_${NAME}_TEST_FILES"
        "${tests}"
    )

endfunction()

function(corekit_add_impl NAME)
    # ----------------------------------------------------------
    # Pass 1: Create ALL targets
    # ----------------------------------------------------------

    cmake_parse_arguments(
        ARG                                                     # prefix
        ""                                                      # options
        ""                                                      # one-value arguments
        "SOURCE_FILES;INCLUDE_PATHS;DEPENDENCIES;TEST_FILES"    # multi-value arguments
        ${ARGN}                                                 # arguments to parse
    )

    make_paths_abs(sources ${ARG_SOURCE_FILES})
    make_paths_abs(includes ${ARG_INCLUDE_PATHS})
    make_paths_abs(tests ${ARG_TEST_FILES})

    # Remember module name.
    set_property(
        GLOBAL APPEND
        PROPERTY COREKIT_IMPL_MODULES
        "${NAME}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_IMPL_${NAME}_SOURCE_FILES"
        "${sources}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_IMPL_${NAME}_INCLUDE_PATHS"
        "${includes}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_IMPL_${NAME}_DEPENDENCIES"
        "${ARG_DEPENDENCIES}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_IMPL_${NAME}_TEST_FILES"
        "${tests}"
    )

endfunction()


function(corekit_finalize)
    find_package(GTest QUIET)

    get_property(
        api_modules
        GLOBAL
        PROPERTY COREKIT_API_MODULES
    )

    get_property(
        impl_modules
        GLOBAL
        PROPERTY COREKIT_IMPL_MODULES
    )

    list(REMOVE_DUPLICATES api_modules)
    list(REMOVE_DUPLICATES impl_modules)

    # ----------------------------------------------------------
    # Pass 1: Create all public API facade targets
    # ----------------------------------------------------------

    foreach(module IN LISTS api_modules)
        set(api_target "corekit-${module}")

        if(NOT TARGET "${api_target}")
            add_library("${api_target}" INTERFACE)
            add_library("corekit::${module}" ALIAS "${api_target}")
        endif()
    endforeach()

    # ----------------------------------------------------------
    # Pass 2: Create all implementation targets
    # ----------------------------------------------------------

    foreach(module IN LISTS impl_modules)
        get_property(
            sources
            GLOBAL
            PROPERTY "COREKIT_IMPL_${module}_SOURCE_FILES"
        )

        set(impl_target "corekit-${module}-impl")

        if(TARGET "${impl_target}")
            continue()
        endif()

        if(sources)
            add_library("${impl_target}" STATIC ${sources})
        else()
            add_library("${impl_target}" INTERFACE)
        endif()

        add_library(
            "corekit::${module}::impl"
            ALIAS
            "${impl_target}"
        )
    endforeach()

    # ----------------------------------------------------------
    # Pass 3: Configure all implementation targets
    # ----------------------------------------------------------

    foreach(module IN LISTS impl_modules)
        get_property(
            sources
            GLOBAL
            PROPERTY "COREKIT_IMPL_${module}_SOURCE_FILES"
        )

        get_property(
            includes
            GLOBAL
            PROPERTY "COREKIT_IMPL_${module}_INCLUDE_PATHS"
        )

        get_property(
            dependencies
            GLOBAL
            PROPERTY "COREKIT_IMPL_${module}_DEPENDENCIES"
        )

        set(impl_target "corekit-${module}-impl")

        if(sources)
            set(usage PUBLIC)
        else()
            set(usage INTERFACE)
        endif()

        if(includes)
            target_include_directories(
                "${impl_target}"
                ${usage}
                ${includes}
            )
        endif()

        foreach(dep IN LISTS dependencies)
            if(dep IN_LIST api_modules)
                target_link_libraries(
                    "${impl_target}"
                    ${usage}
                    "corekit::${dep}"
                )
            elseif(TARGET "${dep}")
                target_link_libraries(
                    "${impl_target}"
                    ${usage}
                    "${dep}"
                )
            elseif("${dep}" MATCHES "^corekit::")
                message(
                    WARNING
                    "Implementation '${module}' references unavailable "
                    "dependency '${dep}'"
                )
            else()
                # External library name, such as pthread, fmt::fmt,
                # Threads::Threads, etc.
                target_link_libraries(
                    "${impl_target}"
                    ${usage}
                    "${dep}"
                )
            endif()
        endforeach()
    endforeach()

    # ----------------------------------------------------------
    # Pass 4: Configure public API facade targets
    # ----------------------------------------------------------

    foreach(module IN LISTS api_modules)
        get_property(
            api_includes
            GLOBAL
            PROPERTY "COREKIT_API_${module}_INCLUDE_PATHS"
        )

        get_property(
            api_dependencies
            GLOBAL
            PROPERTY "COREKIT_API_${module}_DEPENDENCIES"
        )

        set(api_target "corekit-${module}")

        if(api_includes)
            target_include_directories(
                "${api_target}"
                INTERFACE
                ${api_includes}
            )
        endif()

        # Attach this module's concrete implementation.
        if(module IN_LIST impl_modules)
            target_link_libraries(
                "${api_target}"
                INTERFACE
                "corekit::${module}::impl"
            )

            string(TOUPPER "${module}" module_upper)

            target_compile_definitions(
                "${api_target}"
                INTERFACE
                "COREKIT_HAS_${module_upper}=1"
            )
        endif()

        # Attach other public module dependencies.
        foreach(dep IN LISTS api_dependencies)
            if(dep IN_LIST api_modules)
                target_link_libraries(
                    "${api_target}"
                    INTERFACE
                    "corekit::${dep}"
                )
            else()
                message(
                    WARNING
                    "API module '${module}' references unavailable "
                    "module dependency '${dep}'"
                )
            endif()
        endforeach()
    endforeach()

    # ----------------------------------------------------------
    # Pass 5: Create tests after the full target graph is wired
    # ----------------------------------------------------------

    if(GTest_FOUND)
        foreach(module IN LISTS api_modules)
            get_property(
                api_tests
                GLOBAL
                PROPERTY "COREKIT_API_${module}_TEST_FILES"
            )

            if(api_tests)
                add_executable(
                    "corekit-${module}-api-test"
                    ${api_tests}
                )

                target_link_libraries(
                    "corekit-${module}-api-test"
                    PRIVATE
                    "corekit::${module}"
                    GTest::gtest
                    GTest::gtest_main
                )

                include(GoogleTest)
                gtest_discover_tests("corekit-${module}-api-test")

            endif()
        endforeach()

        foreach(module IN LISTS impl_modules)
            get_property(
                impl_tests
                GLOBAL
                PROPERTY "COREKIT_IMPL_${module}_TEST_FILES"
            )

            if(impl_tests)
                add_executable(
                    "corekit-${module}-impl-test"
                    ${impl_tests}
                )

                # Tests intentionally consume the public facade.
                # That verifies that API -> implementation wiring works.
                target_link_libraries(
                    "corekit-${module}-impl-test"
                    PRIVATE
                    "corekit::${module}"
                    GTest::gtest
                    GTest::gtest_main
                )

                include(GoogleTest)
                gtest_discover_tests("corekit-${module}-impl-test")

            endif()
        endforeach()
    endif()
endfunction()