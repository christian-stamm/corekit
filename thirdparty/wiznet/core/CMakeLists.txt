set(CORE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Ethernet
add_library(wiznet-core STATIC
        ${CORE_DIR}/src/socket.c
        ${CORE_DIR}/src/w6300.c
        ${CORE_DIR}/src/wizchip_conf.c
)

target_include_directories(wiznet-core PUBLIC
        ${CORE_DIR}/inc
)