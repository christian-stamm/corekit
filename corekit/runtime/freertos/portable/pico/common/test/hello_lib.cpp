#include <pico/stdlib.h>

#include "corekit/logger.hpp"
#include "corekit/uart.hpp"

using namespace corekit;

LogDevice::Ptr uartdev = std::make_shared<UART>();

int _write(int fd, const void *buf, size_t count) {
    uartdev->writeBulk({reinterpret_cast<const uint8_t *>(buf), count});
    return count;
}

int main() {
    Logging::setLevel(LogLevel::DEBUG);
    uartdev->load();

    Logging::setStream(uartdev);

    {
        Logger logger("TestLogger");
        logger() << "This is a test log message.";
        sleep_ms(1000);

        std::cout << "This is a test log message to std::cout." << std::endl;
        sleep_ms(1000);

        printf("This is a test log message to printf.\n");
        sleep_ms(1000);
    };

    uartdev->unload();

    return 0;
}