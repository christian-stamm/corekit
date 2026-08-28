cmake_minimum_required(VERSION 3.25)

corekit_add_module(asyncdevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/asyncdevice.cpp"
    DEPENDENCIES        "serialdevice"
)

corekit_add_module(uartdevice IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/uartdevice.cpp"
    DEPENDENCIES        "hardware_uart;hardware_gpio;asyncdevice"
)

corekit_add_module(spidevice IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/spidevice.cpp"
    DEPENDENCIES        "hardware_spi;hardware_gpio;asyncdevice"
)