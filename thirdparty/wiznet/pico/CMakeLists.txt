set(PORT_DIR ${CMAKE_CURRENT_LIST_DIR})

# port
add_library(wiznet-inet STATIC
        ${PORT_DIR}/src/wizchip_spi.c
        ${PORT_DIR}/src/wizchip_qspi_pio.c
)

target_compile_definitions(wiznet-inet PUBLIC
        -DWIZNET_SPI_FREQ=${WIZNET_SPI_FREQ}
)

pico_generate_pio_header(wiznet-inet ${PORT_DIR}/src/wizchip_qspi_pio.pio)

target_include_directories(wiznet-inet PUBLIC
        ${PORT_DIR}/inc
)

target_link_libraries(wiznet-inet PUBLIC
        wiznet-core
        pico_stdlib
        hardware_clocks
        hardware_spi
        hardware_dma
        hardware_pio
)