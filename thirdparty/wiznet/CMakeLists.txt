cmake_minimum_required(VERSION 3.25)

add_definitions(-D_WIZCHIP_SPI_SCLK_SPEED=80) 

add_subdirectory(core)
add_subdirectory(pico)