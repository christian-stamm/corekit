#pragma once

#include "corekit/device/driver/serial.hpp"
#include "corekit/utils/memory.hpp"

using namespace corekit::device;
using namespace corekit::utils;
using namespace corekit::system::concurrency;

class TestDevice : public Serial<uint8_t> {
   public:
    TestDevice() : Serial<uint8_t>("TestDevice") {}

    bool read(std::span<uint8_t>& dst) const override {
        if (loopBackBuffer.size() != dst.size()) {
            dst = std::span<uint8_t>(loopBackBuffer.data(),
                                     loopBackBuffer.size());
        }

        std::copy(loopBackBuffer.begin(), loopBackBuffer.end(), dst.begin());
        return true;
    }

    bool write(const std::span<uint8_t>& src) const override {
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