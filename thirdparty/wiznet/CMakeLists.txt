cmake_minimum_required(VERSION 3.25)

set(WIZNET_CHIP 6300)
set(WIZNET_SPI_MODE QSPI_QUAD_MODE)
set(WIZNET_SPI_FREQ 75)

add_subdirectory(core)
add_subdirectory(pico)