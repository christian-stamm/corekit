#pragma once

#include <cstdlib>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

#include "corekit/core.hpp"

namespace corekit {
    namespace structs {

        namespace gpio {

            using Pin   = uint;
            using Group = std::set<Pin>;

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

                    return Pin(first() + math::wrap(index, int(this->size())));
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

        };  // namespace gpio

    };  // namespace structs
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