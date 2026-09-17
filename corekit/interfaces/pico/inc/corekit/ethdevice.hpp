#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
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
            MacAddress  mac     = {};
            IpV4Address address = {};
            SubnetMask  subnet  = {};
            Gateway     gateway = {};
            Dns         dns     = {};
            Port        local   = 0;
    };

    class Device final : public BaseDevice {
        public:

            using Ptr = std::shared_ptr<Device>;

            explicit Device();
            ~Device() override;

            Device(const Device&)            = delete;
            Device(Device&&)                 = delete;
            Device& operator=(const Device&) = delete;
            Device& operator=(Device&&)      = delete;

            bool apply(const Config& config);

            [[nodiscard]]
            bool link_up() const;

            [[nodiscard]]
            bool connected() const;

            [[nodiscard]]
            bool connect_to(Endpoint remote);

            int32_t send(std::span<const uint8_t> payload);
            int32_t recv(std::span<uint8_t> payload);

            friend std::ostream& operator<<(std::ostream& os, const Device& dev) {
                os << dev.status();
                return os;
            }

            friend std::ostream& operator<<(std::ostream& os, const Device::Ptr& dev) {
                os << dev->status();
                return os;
            }

        protected:

            bool on_load() override;
            bool on_unload() override;

        private:

            std::string status() const;

            uint8_t               socket_;
            std::optional<Config> config_;
    };

} // namespace corekit::Eth
