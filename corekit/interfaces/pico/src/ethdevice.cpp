#include "corekit/ethdevice.hpp"

#include <algorithm>
#include <array>
#include <limits>

#include "corekit/check.hpp"

extern "C" {
#include "socket.h"
#include "wizchip_spi.h"
}

namespace corekit::Eth {

    constexpr uint8_t kIpv4Length = 4;

    Device::Device(Config config)
        : BaseDevice("EthW6300")
        , socket_(0)
        , config_(config) { }

    Device::~Device() { }

    bool Device::on_load() {
        wizchip_spi_initialize();
        wizchip_cris_initialize();

        wizchip_reset();
        wizchip_initialize();
        wizchip_check();

        wiz_NetInfo net{
            .lla    = {0xfe, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x08, 0xdc, 0xff, 0xfe, 0x57, 0x57, 0x25},
            .sn6    = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
            .dns6   = {0x20, 0x01, 0x48, 0x60, 0x48, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88},
            .ipmode = NETINFO_STATIC_ALL,
            .dhcp   = NETINFO_STATIC
        };

        std::copy(config_.mac.begin(), config_.mac.end(), net.mac);
        std::copy(config_.address.begin(), config_.address.end(), net.ip);
        std::copy(config_.subnet.begin(), config_.subnet.end(), net.sn);
        std::copy(config_.gateway.begin(), config_.gateway.end(), net.gw);
        std::copy(config_.dns.begin(), config_.dns.end(), net.dns);

        network_initialize(net);

        socket_ = socket(0, Sn_MR_UDP, config_.local, 0);

        return true;
    }

    bool Device::on_unload() { return core::check(close(socket_) == SOCK_OK, RuntimeError("Failed to close socket")); }

    int32_t Device::send_to(std::span<const uint8_t> payload, const Endpoint& remote) {
        core::check(is_loaded(), RuntimeError("Ethernet device is not loaded"));
        core::check(!payload.empty(), RuntimeError("Payload is empty"));
        core::check(
            payload.size() <= std::numeric_limits<uint16_t>::max(),
            RuntimeError("Payload size exceeds maximum allowed size"));

        auto* data = const_cast<uint8_t*>(payload.data());
        auto  addr = remote.address;
        return sendto(socket_, data, static_cast<uint16_t>(payload.size()), addr.data(), remote.port, kIpv4Length);
    }

    int32_t Device::recv_from(std::span<uint8_t> payload, Endpoint& remote) {
        core::check(is_loaded(), RuntimeError("Ethernet device is not loaded"));
        core::check(!payload.empty(), RuntimeError("Payload is empty"));
        core::check(
            payload.size() <= std::numeric_limits<uint16_t>::max(),
            RuntimeError("Payload size exceeds maximum allowed size"));

        IpV4Address addr    = {0, 0, 0, 0};
        Port        port    = 0;
        uint8_t     addrlen = kIpv4Length;

        const int32_t rc =
            recvfrom(socket_, payload.data(), static_cast<uint16_t>(payload.size()), addr.data(), &port, &addrlen);

        core::check(rc >= 0, RuntimeError("Failed to receive data from socket"));
        core::check(addrlen == kIpv4Length, RuntimeError("Received address length is not IPv4"));

        std::copy_n(addr.begin(), kIpv4Length, remote.address.begin());
        remote.port = port;

        return rc;
    }

} // namespace corekit::Eth