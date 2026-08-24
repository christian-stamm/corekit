#pragma once
#include <cuda_runtime.h>
#include <vector_functions.h>

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "corekit/types.hpp"
#include "corekit/utils/assert.hpp"
#include "corekit/utils/logger.hpp"

using uint  = unsigned int;
using uchar = unsigned char;

namespace corekit {
    namespace cuda {

        struct Mat4 {
            Mat4() : Mat4(identity()) {}
            Mat4(const Mat4& other) : rows(other.rows) {}

            static Mat4 identity() {
                Mat4 m;
                m.rows[0] = make_float4(1, 0, 0, 0);
                m.rows[1] = make_float4(0, 1, 0, 0);
                m.rows[2] = make_float4(0, 0, 1, 0);
                m.rows[3] = make_float4(0, 0, 0, 1);
                return m;
            }

            Mat4& operator=(const Mat4& other) {
                if (this != &other) {
                    rows[0] = other.rows[0];
                    rows[1] = other.rows[1];
                    rows[2] = other.rows[2];
                    rows[3] = other.rows[3];
                }
                return *this;
            }

            float4 operator*(const float4& vec) const {
                float4 result;
                result.x = rows[0].x * vec.x + rows[0].y * vec.y +
                           rows[0].z * vec.z + rows[0].w * vec.w;
                result.y = rows[1].x * vec.x + rows[1].y * vec.y +
                           rows[1].z * vec.z + rows[1].w * vec.w;
                result.z = rows[2].x * vec.x + rows[2].y * vec.y +
                           rows[2].z * vec.z + rows[2].w * vec.w;
                result.w = rows[3].x * vec.x + rows[3].y * vec.y +
                           rows[3].z * vec.z + rows[3].w * vec.w;
                return result;
            }

            Mat4 operator*(const Mat4& other) const {
                Mat4 result;

                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        float sum = 0.0f;
                        for (int k = 0; k < 4; ++k) {
                            sum += rows[i][k] * other.rows[k][j];
                        }
                        result.rows[i][j] = sum;
                    }
                }

                return result;
            }

            float& operator()(int row, int col) {
                return get(row, col);
            }

            float& get(int row, int col) {
                return rows[row][col];
            }

            float operator()(int row, int col) const {
                return get(row, col);
            }

            float get(int row, int col) const {
                return rows[row][col];
            }

            void set(int row, int col, float value) {
                rows[row][col] = value;
            }

            Mat4 clone() const {
                return Mat4(*this);
            }

           private:
            float4 rows[4];
        };

    }  // namespace cuda

}  // namespace corekit
