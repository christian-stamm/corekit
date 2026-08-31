cmake_minimum_required(VERSION 3.25)

corekit_add_module(asyncdevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/asyncdevice.cpp"
    DEPENDENCIES        "serialdevice"
)

corekit_add_module(gpiodevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/gpiodevice.cpp"
    DEPENDENCIES        "core;result;task;queue;time;hardware_gpio"
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
    DEPENDENCIES        "hardware_pio;dmadevice;gpiodevice"
)

corekit_add_module(dmadevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/dmadevice.cpp"
    DEPENDENCIES        "hardware_dma;asyncdevice;atomic;core;semaphore;task"
)

corekit_add_module(qspidevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/qspidevice.cpp"
    PIO_FILES           "pio/qspi_quad.pio"
    DEPENDENCIES        "basedevice;piodevice;dmadevice;gpiodevice;atomic;mutex;semaphore;hardware_pio;hardware_gpio;hardware_clocks"
)

corekit_add_module(ethdevice API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/ethdevice.cpp"
    DEPENDENCIES        "basedevice;qspidevice;gpiodevice;mutex;time;wiznet::w6300"
)
