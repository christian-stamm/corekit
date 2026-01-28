#include "corekit/core.hpp"

namespace corekit {
    namespace utils {
        namespace ops {

            template <typename T>
            T div(T dividend, T divisor, T fallback) {
                return divisor ? dividend / divisor : fallback;
            }

            int wrap(int value, int length) {
                if (length == 0) {
                    return 0;
                }

                value %= length;
                value += length;
                value %= length;
                return value;
            }

            // Explicit instantiations (add more if needed)
            template int    div<int>(int, int, int);
            template float  div<float>(float, float, float);
            template double div<double>(double, double, double);

        };  // namespace ops
    };      // namespace utils
};          // namespace corekit
