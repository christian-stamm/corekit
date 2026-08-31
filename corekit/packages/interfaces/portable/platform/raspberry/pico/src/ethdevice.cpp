#include "corekit/ethdevice.hpp"

#include <algorithm>
#include <array>
#include <limits>

#include "corekit/gpiodevice.hpp"
#include "corekit/time.hpp"

extern "C" {
#include "W6300/w6300.h"
#include "socket.h"
#include "wizchip_conf.h"
}

namespace corekit::Eth {

    namespace {
        constexpr uint8_t  kIpv4Length    = 4;
        constexpr uint8_t  kSocketFlags   = SF_IO_NONBLOCK;
        constexpr uint32_t kW6300MaxSckHz = 75'000'000;

        // W6300 quad mode still serializes the 8-bit opcode over IO0. Each
        // opcode bit is therefore expanded into one QSPI nibble; the 16-bit
        // address and dummy cycle are native quad transfers. This matches
        // WIZnet's reference PIO port.
        std::array<uint8_t, 7> make_quad_header(uint8_t opcode, uint16_t address) {
            return {
                static_cast<uint8_t>(((opcode >> 7) & 0x01) << 4 | ((opcode >> 6) & 0x01)),
                static_cast<uint8_t>(((opcode >> 5) & 0x01) << 4 | ((opcode >> 4) & 0x01)),
                static_cast<uint8_t>(((opcode >> 3) & 0x01) << 4 | ((opcode >> 2) & 0x01)),
                static_cast<uint8_t>(((opcode >> 1) & 0x01) << 4 | ((opcode >> 0) & 0x01)),
                static_cast<uint8_t>(address >> 8),
                static_cast<uint8_t>(address & 0xFF),
                0,
            };
        }
    } // namespace

    Device* Device::active_ = nullptr;

    Device::Device(Config config)
        : BaseDevice("EthW6300")
        , config_(config)
        , qspi_(config.bus) { }

    Device::~Device() {
        if (active_ == this) { active_ = nullptr; }
    }

    bool Device::on_load() {
        if (active_ != nullptr && active_ != this) { return false; }
        active_ = this;

        if (!init_transport()) {
            active_ = nullptr;
            return false;
        }

        reset_chip();

        if (!init_chip()) { return false; }

        if (!open_udp_socket()) { return false; }

        return true;
    }

    bool Device::on_unload() {
        close(config_.socket);
        qspi_.unload();

        Gpio::configure(config_.cs_pin, false, false, GPIO_IN);
        Gpio::configure(config_.reset_pin, false, false, GPIO_IN);
        Gpio::configure(config_.irq_pin, false, false, GPIO_IN);

        if (active_ == this) { active_ = nullptr; }
        return true;
    }

    bool Device::init_transport() {
        // The W6300 datasheet specifies 75 MHz maximum SCLK. Keep this guard
        // here even though Qspi::Device is intentionally protocol agnostic.

        if (qspi_.actual_clock_hz() > kW6300MaxSckHz || !qspi_.load()) { return false; }

        if (!Gpio::configure(config_.cs_pin, false, false, GPIO_OUT) || !Gpio::configure(config_.reset_pin, false, false, GPIO_OUT) || !Gpio::configure(config_.irq_pin, true, false, GPIO_IN)) {
            return false;
        }

        Gpio::setPinLevel(config_.cs_pin, true);
        Gpio::setPinLevel(config_.reset_pin, true);

        reg_wizchip_cris_cbfunc(&Device::critical_enter_cb, &Device::critical_exit_cb);
        reg_wizchip_cs_cbfunc(&Device::cs_select_cb, &Device::cs_deselect_cb);
        reg_wizchip_qspi_cbfunc(&Device::qspi_read_cb, &Device::qspi_write_cb);

        return true;
    }

    void Device::reset_chip() {
        Gpio::setPinLevel(config_.reset_pin, false);
        Time::sleep(0.1);
        Gpio::setPinLevel(config_.reset_pin, true);
        Time::sleep(0.1);
    }

    bool Device::init_chip() {
        const uint8_t  cidr0 = WIZCHIP_READ(_CIDR_);
        const uint8_t  cidr1 = WIZCHIP_READ(WIZCHIP_OFFSET_INC(_CIDR_, 1));
        const uint8_t  rtl   = WIZCHIP_READ(_RTL_);
        const uint16_t ver   = getVER();

        // 2 KiB RX + 2 KiB TX per socket = 16 KiB in each W6300 direction.
        uint8_t memory[2][8] = {
            {2, 2, 2, 2, 2, 2, 2, 2},
            {2, 2, 2, 2, 2, 2, 2, 2},
        };
        if (wizchip_init(memory[0], memory[1]) != 0) { return false; }

        wiz_NetInfo net{};
        std::copy(config_.network.mac.begin(), config_.network.mac.end(), net.mac);
        std::copy(config_.network.address.octets.begin(), config_.network.address.octets.end(), net.ip);
        std::copy(config_.network.subnet.octets.begin(), config_.network.subnet.octets.end(), net.sn);
        std::copy(config_.network.gateway.octets.begin(), config_.network.gateway.octets.end(), net.gw);
        std::copy(config_.network.dns.octets.begin(), config_.network.dns.octets.end(), net.dns);
        net.dhcp        = NETINFO_STATIC;

        uint8_t syslock = SYS_NET_LOCK;
        if (ctlwizchip(CW_SYS_UNLOCK, &syslock) != 0) { return false; }
        wizchip_setnetinfo(&net);

        return getCIDR() == 0x6300;
    }

    bool Device::open_udp_socket() {
        close(config_.socket);
        const int8_t rc = socket(config_.socket, Sn_MR_UDP, config_.local_port, kSocketFlags);
        return rc == static_cast<int8_t>(config_.socket);
    }

    bool Device::link_up() const {
        if (!is_loaded()) { return false; }
        uint8_t link = PHY_LINK_OFF;
        return ctlwizchip(CW_GET_PHYLINK, &link) == 0 && link == PHY_LINK_ON;
    }

    int32_t Device::send_to(std::span<const uint8_t> payload, const Endpoint& remote) {
        if (!is_loaded() || payload.empty() || payload.size() > std::numeric_limits<uint16_t>::max()) { return 0; }

        auto* data = const_cast<uint8_t*>(payload.data());
        auto  addr = remote.address.octets;
        return sendto(config_.socket, data, static_cast<uint16_t>(payload.size()), addr.data(), remote.port, kIpv4Length);
    }

    int32_t Device::receive_from(std::span<uint8_t> payload, Endpoint& remote) {
        if (!is_loaded() || payload.empty() || payload.size() > std::numeric_limits<uint16_t>::max()) { return 0; }

        uint16_t                port    = 0;
        uint8_t                 addrlen = kIpv4Length;
        std::array<uint8_t, 16> addr{};
        const int32_t           rc = recvfrom(config_.socket, payload.data(), static_cast<uint16_t>(payload.size()), addr.data(), &port, &addrlen);
        if (rc > 0 && addrlen >= kIpv4Length) {
            std::copy_n(addr.begin(), kIpv4Length, remote.address.octets.begin());
            remote.port = port;
        }
        return rc;
    }

    void Device::chip_select(bool selected) { Gpio::setPinLevel(config_.cs_pin, !selected); }

    bool Device::qspi_write_frame(uint8_t opcode, uint16_t address, std::span<const uint8_t> payload) {
        const auto header = make_quad_header(opcode, address);
        return qspi_.write(header, payload);
    }

    bool Device::qspi_read_frame(uint8_t opcode, uint16_t address, std::span<uint8_t> payload) {
        const auto header = make_quad_header(opcode, address);
        return qspi_.read(header, payload);
    }

    void Device::critical_enter_cb() {
        if (active_) { active_->transaction_mutex_.lock(); }
    }

    void Device::critical_exit_cb() {
        if (active_) { active_->transaction_mutex_.unlock(); }
    }

    void Device::cs_select_cb() {
        if (active_) { active_->chip_select(true); }
    }

    void Device::cs_deselect_cb() {
        if (active_) { active_->chip_select(false); }
    }

    void Device::qspi_read_cb(uint8_t opcode, uint16_t address, uint8_t* data, uint16_t length) {
        if (active_) { active_->qspi_read_frame(opcode, address, std::span<uint8_t>(data, length)); }
    }

    void Device::qspi_write_cb(uint8_t opcode, uint16_t address, uint8_t* data, uint16_t length) {
        if (active_) { active_->qspi_write_frame(opcode, address, std::span<const uint8_t>(data, length)); }
    }

} // namespace corekit::Eth
