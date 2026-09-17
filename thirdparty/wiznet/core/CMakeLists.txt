set(CORE_DIR $ENV{WIZNET_DRIVER_PATH})

# Ethernet
add_library(wiznet-sock STATIC
        ${CORE_DIR}/Ethernet/socket.c
        ${CORE_DIR}/Ethernet/wizchip_conf.c
)

target_compile_definitions(wiznet-sock PUBLIC
        -D_WIZCHIP_=W${WIZNET_CHIP}
)

target_include_directories(wiznet-sock PUBLIC
        ${CORE_DIR}/Ethernet
)


add_library(wiznet-core STATIC
        ${CORE_DIR}/Ethernet/W${WIZNET_CHIP}/w${WIZNET_CHIP}.c
)

target_compile_definitions(wiznet-core PUBLIC
        -D_WIZCHIP_QSPI_MODE_=${WIZNET_SPI_MODE}
)

target_include_directories(wiznet-core PUBLIC
        ${CORE_DIR}/Ethernet/W${WIZNET_CHIP}
)

target_link_libraries(wiznet-core PUBLIC
        wiznet-sock
)