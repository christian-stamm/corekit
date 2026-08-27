#include "corekit/logger.hpp"

int main() {
    {
        corekit::Logger logger("Hello");
        logger.info() << "Hello, World!";
        logger.debug() << "Hello, World!";
        logger.warn() << "Hello, World!";
        logger.error() << "Hello, World!";
        logger.fatal() << "Hello, World!";
    }

    return 0;
}