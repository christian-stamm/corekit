#pragma once

#include <cstdlib>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "corekit/types.hpp"
#include "corekit/utils/math.hpp"

namespace corekit {
    namespace device {

        namespace GPIO {

            using namespace corekit::math;
            using namespace corekit::types;
            using namespace corekit::types::GPIO;

            struct Range : public Group {
                Range(Pin base = 0, size_t count = 2) {
                    for (Pin pin = 0; pin < count; ++pin) {
                        this->insert(base + pin);
                    }
                }

                Range subset(Pin shift, size_t length) const {
                    if (this->size() < (shift + length)) {
                        throw std::out_of_range("Range subset out of bounds");
                    }

                    return Range(at(first() + shift), length);
                }

                Pin at(int index) const {
                    if (int(this->size()) <= std::abs(index)) {
                        throw std::out_of_range("Range index out of bounds");
                    }

                    return Pin(first() + ops::wrap(index, int(this->size())));
                }

                Pin operator[](int index) const {
                    return at(index);
                }

                Pin first() const {
                    return *this->begin();
                }

                Pin last() const {
                    return *this->rbegin();
                }
            };

        };  // namespace GPIO

    };  // namespace device
};  // namespace corekit

namespace nlohmann {

    inline void from_json(const ordered_json&            j,
                          corekit::structs::gpio::Range& r) {
        const uint base  = j.value<uint>("base", 0);
        const uint count = j.value<uint>("count", 0);

        r = corekit::structs::gpio::Range(base, count);
    }

    inline void to_json(ordered_json&                        j,
                        const corekit::structs::gpio::Range& r) {
        j = ordered_json{
            {"base", r.first()},
            {"count", r.size()},
        };
    }

};  // namespace nlohmann