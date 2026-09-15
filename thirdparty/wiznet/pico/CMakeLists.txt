set(PORT_DIR ${CMAKE_CURRENT_LIST_DIR})

add_definitions(-DDEVICE_BOARD_NAME=W6300_EVB_PICO2)
add_definitions(-D_WIZCHIP_QSPI_MODE_=QSPI_QUAD_MODE)
add_definitions(-D_WIZCHIP_SPI_SCLK_SPEED=44) # SPEED MHz

# port
add_library(wiznet-inet STATIC
        ${PORT_DIR}/src/wizchip_spi.c
        ${PORT_DIR}/src/wizchip_qspi_pio.c
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