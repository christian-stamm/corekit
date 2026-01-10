#pragma once
//
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
//
#include <cmath>

namespace corekit {
    namespace structs {

        struct Vec2 : public glm::vec2 {
            using glm::vec2::vec2;

            Vec2(float width = 0, float height = 0)
                : glm::vec2(width, height) {}

            explicit operator bool() const {
                return x != 0 && y != 0;
            }

            bool operator==(const Vec2& other) const {
                return x == other.x && y == other.y;
            }

            Vec2 operator*(const Vec2& other) const {
                return Vec2{x * other.x, y * other.y};
            }

            Vec2 operator/(const Vec2& other) const {
                return Vec2{x / other.x, y / other.y};
            }
        };

    };  // namespace structs
};  // namespace corekit

namespace nlohmann {

    inline void from_json(const ordered_json& j, corekit::structs::Vec2& s) {
        s.x = j.at("width");
        s.y = j.at("height");
    }

    inline void to_json(ordered_json& j, const corekit::structs::Vec2& s) {
        j = ordered_json{
            {"width", s.x},   //
            {"height", s.y},  //
        };
    };

};  // namespace nlohmann
