cmake_minimum_required(VERSION 3.25)

corekit_add_module(asyncdevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/asyncdevice.cpp"
    DEPENDENCIES        "serialdevice"
)

corekit_add_module(gpiodevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/gpiodevice.cpp"
    DEPENDENCIES        "hardware_gpio"
)

corekit_add_module(uartdevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/uartdevice.cpp"
    DEPENDENCIES        "hardware_uart;asyncdevice;gpiodevice"
)

corekit_add_module(spidevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/spidevice.cpp"
    DEPENDENCIES        "hardware_spi;asyncdevice;gpiodevice"
)

corekit_add_module(piodevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/piodevice.cpp"
    DEPENDENCIES        "hardware_pio;asyncdevice;gpiodevice"
)

corekit_add_module(dmadevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/dmadevice.cpp"
    DEPENDENCIES        "hardware_dma;asyncdevice"
)