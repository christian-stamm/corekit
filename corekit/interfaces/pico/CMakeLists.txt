cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_RP2XXX_MODULES)
    return()
endif()

corekit_add_module(asyncdevice
    INCLUDES     "inc"
    SOURCES      "src/asyncdevice.cpp"
    DEPENDENCIES "serialdevice;check"
)

corekit_add_module(gpiodevice
    INCLUDES     "inc"
    SOURCES      "src/gpiodevice.cpp"
    DEPENDENCIES "atomic;check;hardware_gpio;math;semaphore;task"
)

corekit_add_module(uartdevice
    INCLUDES     "inc"
    SOURCES      "src/uartdevice.cpp"
    DEPENDENCIES "asyncdevice;gpiodevice;hardware_uart"
)

corekit_add_module(spidevice
    INCLUDES     "inc"
    SOURCES      "src/spidevice.cpp"
    DEPENDENCIES "asyncdevice;gpiodevice;hardware_spi"
)

corekit_add_module(piodevice
    INCLUDES     "inc"
    SOURCES      "src/piodevice.cpp"
    DEPENDENCIES "asyncdevice;gpiodevice;hardware_pio"
)

corekit_add_module(dmadevice
    INCLUDES     "inc"
    SOURCES      "src/dmadevice.cpp"
    DEPENDENCIES "asyncdevice;atomic;hardware_dma;math;semaphore;task"
)

corekit_add_module(ethdevice
    INCLUDES     "inc"
    SOURCES      "src/ethdevice.cpp"
    DEPENDENCIES "basedevice;gpiodevice;mutex;wiznet-inet"
)
