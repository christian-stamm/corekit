#pragma once
#include <cstdint>
#include <type_traits>

namespace corekit::math {

    inline int64_t wrap(int64_t index, const int64_t& length) {
        index %= length;
        index += length;
        index %= length;
        return index;
    }

    inline bool isPow2(uint64_t value) {
        return (value & (value - 1)) == 0;
    }

}  // namespace corekit::math
