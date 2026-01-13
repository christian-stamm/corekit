#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <nlohmann/json.hpp>

#include "corekit/types.hpp"

namespace corekit {
    namespace math {

        using Vec2 = Eigen::Vector2f;
        using Vec3 = Eigen::Vector3f;

        // Element-wise division operators for Eigen vectors
        inline Vec2 operator/(const Vec2& lhs, const Vec2& rhs) {
            return lhs.cwiseQuotient(rhs);
        }

        inline Vec3 operator/(const Vec3& lhs, const Vec3& rhs) {
            return lhs.cwiseQuotient(rhs);
        }

        // Element-wise division operators for Eigen vectors
        inline Vec2 operator*(const Vec2& lhs, const Vec2& rhs) {
            return lhs.cwiseProduct(rhs);
        }

        inline Vec3 operator*(const Vec3& lhs, const Vec3& rhs) {
            return lhs.cwiseProduct(rhs);
        }

        namespace ops {

            template <typename T>
            T div(T dividend, T divisor, T fallback);

            int wrap(int value, int length);

        };  // namespace ops

    };  // namespace math
};      // namespace corekit

namespace nlohmann {

    using namespace corekit::types;
    using namespace corekit::math;

    inline void from_json(const JsonMap& j, Vec2& s) {
        s.x() = j.value("width", 0.f);
        s.y() = j.value("height", 0.f);
    }

    inline void to_json(JsonMap& j, const Vec2& s) {
        j = JsonMap{
            {"width", s.x()},   //
            {"height", s.y()},  //
        };
    };

};  // namespace nlohmann
