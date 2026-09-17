#include "corekit/ethdevice.hpp"

#include <algorithm>
#include <format>
#include <limits>

#include "corekit/check.hpp"
#include "corekit/error.hpp"
#include "corekit/logger.hpp"
#include "corekit/time.hpp"

extern "C" {
#include "socket.h"
#include "wizchip_spi.h"
}

namespace corekit::Eth {

    constexpr uint8_t kMaxBufSize = 32;
    constexpr uint8_t kIpv4Length = 4;

    Device::Device()
        : BaseDevice("ETH")
        , socket_(0)
        , config_(std::nullopt) { }

    Device::~Device() { }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    bool Device::apply(const Config &config) {
        core::check(!is_loaded(), RuntimeError("Cannot configure Ethernet device while loaded"));

        config_ = config;

        return core::ok();
    }

    // -------------------------------------------------------------------------
    // Device lifecycle
    // -------------------------------------------------------------------------

    bool Device::on_load() {
        core::check(config_.has_value(), RuntimeError("Ethernet configuration is not set"));

        const auto &config = *config_;

        wizchip_spi_initialize();
        wizchip_cris_initialize();

        wizchip_reset();
        wizchip_initialize();

        uint8_t memsize[2][8] = {
            {kMaxBufSize, 0, 0, 0, 0, 0, 0, 0},
            {kMaxBufSize, 0, 0, 0, 0, 0, 0, 0}
        };

        core::check(ctlwizchip(CW_INIT_WIZCHIP, memsize) != -1, RuntimeError("WIZchip buffer initialization failed"));

        wizchip_check();

        wiz_NetInfo net{
            .lla = {0xfe, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x08, 0xdc, 0xff, 0xfe, 0x57, 0x57, 0x25},

            .sn6 = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},

            .dns6 = {0x20, 0x01, 0x48, 0x60, 0x48, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88},

            .ipmode = NETINFO_STATIC_ALL,
            .dhcp   = NETINFO_STATIC
        };

        std::copy(config.mac.begin(), config.mac.end(), net.mac);
        std::copy(config.address.begin(), config.address.end(), net.ip);
        std::copy(config.subnet.begin(), config.subnet.end(), net.sn);
        std::copy(config.gateway.begin(), config.gateway.end(), net.gw);
        std::copy(config.dns.begin(), config.dns.end(), net.dns);

        network_initialize(net);

        core::check(link_up(), RuntimeError("Ethernet link is down"));

        const int8_t rc = socket(socket_, Sn_MR_TCP, config.local, 0);

        core::check(rc == socket_, RuntimeError("Failed to create TCP socket"));

        return core::ok();
    }

    bool Device::on_unload() {
        return core::check(close(socket_) == SOCK_OK, RuntimeError("Failed to close TCP socket"));
    }

    // -------------------------------------------------------------------------
    // Connection
    // -------------------------------------------------------------------------

    bool Device::link_up() const {
        uint8_t status = PHY_LINK_OFF;

        ctlwizchip(CW_GET_PHYLINK, &status);

        return status == PHY_LINK_ON;
    }

    bool Device::connect_to(Endpoint remote) {
        Logger logger("Eth");

        core::check(is_loaded(), RuntimeError("Ethernet device is not loaded"));

        core::check(link_up(), RuntimeError("Ethernet link is down"));

        logger.info() << std::format(
            "Connecting to {}.{}.{}.{}:{}",
            remote.address[0],
            remote.address[1],
            remote.address[2],
            remote.address[3],
            remote.port);

        logger.info() << std::format("Socket state before connect: 0x{:02X}", getSn_SR(socket_));

        const int8_t rc = ::connect(socket_, remote.address.data(), remote.port, kIpv4Length);

        logger.info() << std::format("connect rc={}, socket state=0x{:02X}", rc, getSn_SR(socket_));

        core::check(rc == SOCK_OK, RuntimeError("Failed to connect to TCP server"));

        return core::ok();
    }

    bool Device::connected() const {
        if (!is_loaded()) { return false; }

        return getSn_SR(socket_) == SOCK_ESTABLISHED;
    }

    // -------------------------------------------------------------------------
    // TCP I/O
    // -------------------------------------------------------------------------

    int32_t Device::send(std::span<const uint8_t> payload) {
        core::check(is_loaded(), RuntimeError("Ethernet device is not loaded"));

        core::check(connected(), RuntimeError("TCP socket is not connected"));

        core::check(!payload.empty(), RuntimeError("Payload is empty"));

        core::check(
            payload.size() <= std::numeric_limits<uint16_t>::max(),
            RuntimeError("Payload size exceeds maximum allowed size"));

        const int32_t rc =
            ::send(socket_, const_cast<uint8_t *>(payload.data()), static_cast<uint16_t>(payload.size()));

        core::check(rc >= 0, RuntimeError("Failed to send TCP data"));

        return rc;
    }

    int32_t Device::recv(std::span<uint8_t> payload) {
        core::check(is_loaded(), RuntimeError("Ethernet device is not loaded"));

        core::check(connected(), RuntimeError("TCP socket is not connected"));

        core::check(!payload.empty(), RuntimeError("Payload is empty"));

        core::check(
            payload.size() <= std::numeric_limits<uint16_t>::max(),
            RuntimeError("Payload size exceeds maximum allowed size"));

        const int32_t rc = ::recv(socket_, payload.data(), static_cast<uint16_t>(payload.size()));

        core::check(rc >= 0, RuntimeError("Failed to receive TCP data"));

        return rc;
    }

    // -------------------------------------------------------------------------
    // Status
    // -------------------------------------------------------------------------

    std::string Device::status() const {
        uint8_t     chip_id[8] = {};
        wiz_NetInfo net        = {};

        ctlnetwork(CN_GET_NETINFO, &net);
        ctlwizchip(CW_GET_ID, chip_id);

        return std::format(
            "==========================================================\n" " {} network configuration\n\n" " MAC         : {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}\n" " IP          : {}.{}.{}.{}\n" " Subnet Mask : {}.{}.{}.{}\n" " Gateway     : {}.{}.{}.{}\n" " DNS         : {}.{}.{}.{}\n" " Link        : {}\n" " TCP         : {}\n" "==========================================================\n\n",

            reinterpret_cast<const char *>(chip_id),

            net.mac[0],
            net.mac[1],
            net.mac[2],
            net.mac[3],
            net.mac[4],
            net.mac[5],

            net.ip[0],
            net.ip[1],
            net.ip[2],
            net.ip[3],

            net.sn[0],
            net.sn[1],
            net.sn[2],
            net.sn[3],

            net.gw[0],
            net.gw[1],
            net.gw[2],
            net.gw[3],

            net.dns[0],
            net.dns[1],
            net.dns[2],
            net.dns[3],

            link_up() ? "UP" : "DOWN",
            connected() ? "CONNECTED" : "DISCONNECTED");
    }

} // namespace corekit::Eth