#pragma once

#include "corekit/utils/device.hpp"
#include "corekit/utils/memory.hpp"

using namespace corekit::system;
using namespace corekit::utils;

class TestDevice : public Device {
   public:
    TestDevice() : Device("TestDevice") {}

    bool read(std::span<uint8_t>& dst) const {
        if (loopBackBuffer.size() != dst.size()) {
            dst = std::span<uint8_t>(loopBackBuffer.data(),
                                     loopBackBuffer.size());
        }

        std::copy(loopBackBuffer.begin(), loopBackBuffer.end(), dst.begin());
        return true;
    }

    bool write(const std::span<uint8_t>& src) const {
        if (loopBackBuffer.size() != src.size()) {
            loopBackBuffer = Memory<uint8_t>(src.size(), 0);
        }

        std::copy(src.begin(), src.end(), loopBackBuffer.data());
        return true;
    }

   protected:
    virtual bool prepare() override {
        loopBackBuffer.fill(0);
        return true;
    }

    virtual bool cleanup() override {
        loopBackBuffer.fill(0);
        return true;
    }

    mutable Memory<uint8_t> loopBackBuffer;
};