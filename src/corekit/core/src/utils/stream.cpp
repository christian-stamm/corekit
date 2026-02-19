#include "corekit/utils/stream.hpp"

namespace corekit {
    namespace utils {

        std::streambuf::int_type LogBuffer::overflow(
            std::streambuf::int_type c) {
            if (c != EOF) {
                std::cout.put(static_cast<char>(c));
            }

            return c;
        }

        std::streamsize LogBuffer::xsputn(const char*     s,
                                          std::streamsize count) {
            std::cout.write(s, count);
            return count;
        }

        Logstream::Logstream(const std::string& prefix)
            : std::ostream(&buffer)
            , lock(mutex) {
            std::cout << prefix;
        }

        Logstream::~Logstream() {
            std::cout << std::endl << std::flush;
        }

    };  // namespace utils
};      // namespace corekit
