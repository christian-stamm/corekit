#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>

#include "corekit/basedevice.hpp"

namespace corekit::Eth {

    template <std::size_t N>
    using Octet = std::array<uint8_t, N>;

    using IpV4Address = Octet<4>;
    using IpV6Address = Octet<16>;
    using MacAddress  = Octet<6>;
    using SubnetMask  = Octet<4>;
    using Gateway     = Octet<4>;
    using Dns         = Octet<4>;
    using Port        = uint16_t;

    struct Endpoint {
            IpV4Address address = {0, 0, 0, 0};
            Port        port    = 0;
    };

    struct Config {
            MacAddress  mac     = {0x02, 0x00, 0x00, 0x00, 0x00, 0x01};
            IpV4Address address = {192, 168, 1, 2};
            SubnetMask  subnet  = {255, 255, 255, 0};
            Gateway     gateway = {192, 168, 0, 1};
            Dns         dns     = {0, 0, 0, 0};
            Port        local   = 5000;
    };

    class Device final : public BaseDevice {
        public:

            using Ptr = std::shared_ptr<Device>;

            explicit Device(Config config = {});
            ~Device() override;

            Device(const Device&)            = delete;
            Device(Device&&)                 = delete;
            Device& operator=(const Device&) = delete;
            Device& operator=(Device&&)      = delete;

            [[nodiscard]]
            const Config& config() const noexcept {
                return config_;
            }

            [[nodiscard]]
            bool link_up() const;

            int32_t send_to(std::span<const uint8_t> payload, const Endpoint& remote);
            int32_t recv_from(std::span<uint8_t> payload, Endpoint& remote);

        protected:

            bool on_load() override;
            bool on_unload() override;

        private:

            uint8_t socket_;
            Config  config_;
    };

} // namespace corekit::Eth
