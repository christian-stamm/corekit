#include "corekit/qspidevice.hpp"

#include <hardware/clocks.h>
#include <hardware/gpio.h>
#include <hardware/pio.h>

#include <algorithm>
#include <limits>
#include <mutex>

#include "corekit/check.hpp"
#include "corekit/error.hpp"
#include "qspi_quad.pio.h"

namespace corekit::Qspi {

    namespace {
        constexpr uint32_t kNibblesPerByte = 2;

        uint64_t pin_mask(const Gpio::Range& pins) { return pins.mask(); }

        uint32_t pio_pin_mask(PIO block, const Gpio::Range& pins) {
            const uint base = pio_get_gpio_base(block);
            if (pins.lower() < base || pins.upper() >= base + 32) { return 0; }
            return static_cast<uint32_t>(((uint64_t{1} << pins.count()) - 1) << (pins.lower() - base));
        }
    } // namespace

    class Device::Engine final : public Pio::Node {
        public:

            Engine(const Config& config, Pio::Program::Ptr program)
                : Pio::Node(config.pio, std::move(program))
                , config_(config) { }

            [[nodiscard]]
            uint base() const noexcept {
                return base_;
            }

        protected:

            bool build_launch_conf(Pio::LaunchConf& conf) override {
                conf.autostart   = false;
                conf.output_pins = config_.data;
                conf.input_pins  = config_.data;
                conf.set_pins    = config_.data;
                conf.side_pins   = Gpio::Range(config_.clock_pin, 1);
                return true;
            }

            bool build_node_conf(Pio::NodeConf& conf, uint base) override {
                base_               = base;
                conf                = corekit_qspi_quad_program_get_default_config(base);

                const float divider = std::max(1.0f, static_cast<float>(clock_get_hz(clk_sys)) / (2.0f * static_cast<float>(config_.clock_hz)));
                sm_config_set_clkdiv(&conf, divider);
                sm_config_set_out_pins(&conf, config_.data.lower(), config_.data.count());
                sm_config_set_in_pins(&conf, config_.data.lower());
                sm_config_set_set_pins(&conf, config_.data.lower(), config_.data.count());
                sm_config_set_sideset(&conf, 1, false, false);
                sm_config_set_sideset_pins(&conf, config_.clock_pin);
                sm_config_set_in_shift(&conf, false, true, 8);
                sm_config_set_out_shift(&conf, false, true, 8);
                return true;
            }

        private:

            Config config_;
            uint   base_{0};
    };

    Device::Device(Config config)
        : BaseDevice("QspiPio")
        , config_(config)
        , prefix_dma_(Dma::Device::request_unused())
        , payload_dma_(Dma::Device::request_unused())
        , rx_dma_(Dma::Device::request_unused()) { }

    Device::~Device() {
        if (is_loaded()) { unload(); }
    }

    uint32_t Device::actual_clock_hz() const noexcept {
        if (config_.clock_hz == 0) { return 0; }
        const float divider = std::max(1.0f, static_cast<float>(clock_get_hz(clk_sys)) / (2.0f * static_cast<float>(config_.clock_hz)));
        return static_cast<uint32_t>(static_cast<float>(clock_get_hz(clk_sys)) / (2.0f * divider));
    }

    bool Device::validate_config() const {
        if (config_.pio == nullptr || config_.data.count() != 4 || config_.clock_hz == 0) { return false; }

        const uint64_t data_mask = pin_mask(config_.data);
        if ((data_mask & (uint64_t{1} << config_.clock_pin)) != 0) { return false; }
        return true;
    }

    bool Device::on_load() {
        corecheck(validate_config(), InvalidArgumentError("Invalid QSPI PIO configuration"));

        program_                 = std::make_shared<Pio::Program>(corekit_qspi_quad_program);

        engine_                  = std::make_shared<Engine>(config_, program_);

        const bool engine_loaded = engine_->load();

        corecheck(engine_loaded, RuntimeError("Failed to load QSPI PIO engine"));

        const bool prefix_ok  = prefix_dma_->load();

        const bool payload_ok = payload_dma_->load();

        const bool rx_ok      = rx_dma_->load();

        corecheck(prefix_ok && payload_ok && rx_ok, RuntimeError("Failed to allocate QSPI DMA channels"));

        corecheck(engine_->load(), RuntimeError("Failed to load QSPI PIO engine"));
        corecheck(prefix_dma_->load() && payload_dma_->load() && rx_dma_->load(), RuntimeError("Failed to allocate QSPI DMA channels"));

        const gpio_function_t pio_func = static_cast<gpio_function_t>(static_cast<uint>(GPIO_FUNC_PIO0) + pio_get_index(config_.pio));

        for (const Gpio::Pin pin : config_.data.pins()) {
            corecheck(
                Gpio::configure(pin, false, config_.pull_down, GPIO_IN, GPIO_OVERRIDE_NORMAL, pio_func, config_.slew_rate, config_.drive_strength),
                RuntimeError("Failed to configure QSPI data pin"));
            gpio_set_input_hysteresis_enabled(pin, true);
        }

        corecheck(
            Gpio::configure(config_.clock_pin, false, config_.pull_down, GPIO_OUT, GPIO_OVERRIDE_NORMAL, pio_func, config_.slew_rate, config_.drive_strength),
            RuntimeError("Failed to configure QSPI clock pin"));

        if (config_.bypass_input_sync) { hw_set_bits(&config_.pio->input_sync_bypass, pio_pin_mask(config_.pio, config_.data)); }

        rx_dma_->setChannelIRQ([this](int) { complete(true); });
        return true;
    }

    bool Device::on_unload() {
        cancel();

        if (rx_dma_) {
            rx_dma_->setChannelIRQ(nullptr);
            rx_dma_->unload();
        }
        if (payload_dma_) { payload_dma_->unload(); }
        if (prefix_dma_) { prefix_dma_->unload(); }
        if (engine_) { engine_->unload(); }

        if (config_.bypass_input_sync && config_.pio != nullptr) { hw_clear_bits(&config_.pio->input_sync_bypass, pio_pin_mask(config_.pio, config_.data)); }

        engine_.reset();
        program_.reset();
        return true;
    }

    void Device::quiesce() {
        if (!engine_) { return; }

        pio_sm_set_enabled(engine_->block, engine_->node, false);
        pio_sm_clear_fifos(engine_->block, engine_->node);
        pio_sm_restart(engine_->block, engine_->node);
        pio_sm_exec(engine_->block, engine_->node, pio_encode_mov(pio_pins, pio_null));
    }

    void Device::cancel() {
        std::lock_guard<Mutex> lock(state_mutex_);

        if (prefix_dma_) { prefix_dma_->kill(); }
        if (payload_dma_) { payload_dma_->kill(); }
        if (rx_dma_) { rx_dma_->kill(); }
        quiesce();

        completion_ = {};
        operation_  = Operation::None;
        busy_.store(false);
    }

    bool Device::prepare_write(std::size_t byte_count) {
        if (!engine_ || byte_count == 0 || byte_count > (std::numeric_limits<uint32_t>::max() / kNibblesPerByte)) { return false; }

        PIO        block = engine_->block;
        const uint sm    = engine_->node;
        const uint base  = engine_->base();

        pio_sm_set_enabled(block, sm, false);
        pio_sm_set_wrap(block, sm, base + corekit_qspi_quad_offset_write_bits, base + corekit_qspi_quad_offset_write_bits_end - 1);
        pio_sm_clear_fifos(block, sm);
        pio_sm_set_consecutive_pindirs(block, sm, config_.data.lower(), config_.data.count(), true);
        pio_sm_restart(block, sm);
        pio_sm_clkdiv_restart(block, sm);

        pio_sm_put(block, sm, static_cast<uint32_t>(byte_count * kNibblesPerByte - 1));
        pio_sm_exec(block, sm, pio_encode_out(pio_x, 32));
        pio_sm_exec(block, sm, pio_encode_jmp(base + corekit_qspi_quad_offset_write_bits));
        return true;
    }

    bool Device::prepare_read(std::size_t prefix_bytes, std::size_t payload_bytes) {
        if (!engine_ || prefix_bytes == 0 || payload_bytes == 0 || prefix_bytes > (std::numeric_limits<uint32_t>::max() / kNibblesPerByte) || payload_bytes > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        PIO        block = engine_->block;
        const uint sm    = engine_->node;
        const uint base  = engine_->base();

        pio_sm_set_enabled(block, sm, false);
        pio_sm_set_wrap(block, sm, base + corekit_qspi_quad_offset_read_command_bits, base + corekit_qspi_quad_offset_read_bits_end - 1);
        pio_sm_clear_fifos(block, sm);
        pio_sm_set_consecutive_pindirs(block, sm, config_.data.lower(), config_.data.count(), true);
        pio_sm_restart(block, sm);
        pio_sm_clkdiv_restart(block, sm);

        pio_sm_put(block, sm, static_cast<uint32_t>(prefix_bytes * kNibblesPerByte - 1));
        pio_sm_exec(block, sm, pio_encode_out(pio_x, 32));
        pio_sm_put(block, sm, static_cast<uint32_t>(payload_bytes - 1));
        pio_sm_exec(block, sm, pio_encode_out(pio_y, 32));
        pio_sm_exec(block, sm, pio_encode_jmp(base + corekit_qspi_quad_offset_read_command_bits));
        return true;
    }

    bool Device::configure_tx(Dma::Device& dma, std::span<const uint8_t> source, int chain_to) {
        if (source.empty()) { return false; }

        auto transfer = std::make_shared<Dma::Transfer>(
            dma.channel,
            source.data(),
            Dma::AddrUpdt::DMA_ADDRESS_UPDATE_INCREMENT,
            engine_->writer.addr,
            Dma::AddrUpdt::DMA_ADDRESS_UPDATE_NONE,
            static_cast<uint32_t>(source.size()),
            0,
            Dma::XferSize::DMA_SIZE_8,
            engine_->writer.dreq,
            false,
            true,
            false,
            chain_to);
        return dma.configure(std::move(transfer));
    }

    bool Device::configure_rx(std::span<uint8_t> target) {
        if (target.empty()) { return false; }

        auto transfer = std::make_shared<Dma::Transfer>(
            rx_dma_->channel,
            engine_->reader.addr,
            Dma::AddrUpdt::DMA_ADDRESS_UPDATE_NONE,
            target.data(),
            Dma::AddrUpdt::DMA_ADDRESS_UPDATE_INCREMENT,
            static_cast<uint32_t>(target.size()),
            0,
            Dma::XferSize::DMA_SIZE_8,
            engine_->reader.dreq,
            false,
            true,
            false);
        return rx_dma_->configure(std::move(transfer));
    }

    bool Device::configure_write_completion() {
        auto transfer = std::make_shared<Dma::Transfer>(
            rx_dma_->channel,
            engine_->reader.addr,
            Dma::AddrUpdt::DMA_ADDRESS_UPDATE_NONE,
            &write_completion_token_,
            Dma::AddrUpdt::DMA_ADDRESS_UPDATE_NONE,
            1,
            0,
            Dma::XferSize::DMA_SIZE_32,
            engine_->reader.dreq);
        return rx_dma_->configure(std::move(transfer));
    }

    bool Device::begin_write(std::span<const uint8_t> prefix, std::span<const uint8_t> payload, Completion completion) {
        std::lock_guard<Mutex> lock(state_mutex_);

        if (!is_loaded() || busy_.load() || prefix.empty()) { return false; }

        const std::size_t total = prefix.size() + payload.size();
        if (!prepare_write(total) || !configure_write_completion()) {
            quiesce();
            return false;
        }

        if (!payload.empty()) {
            if (!configure_tx(*payload_dma_, payload) || !configure_tx(*prefix_dma_, prefix, static_cast<int>(payload_dma_->channel))) {
                quiesce();
                return false;
            }
        } else if (!configure_tx(*prefix_dma_, prefix)) {
            quiesce();
            return false;
        }

        completion_ = std::move(completion);
        operation_  = Operation::Write;
        busy_.store(true);

        // Arm completion before producing TX data. Payload DMA is chain-started
        // by prefix DMA, giving true scatter/gather without a bounce-buffer
        // copy.
        rx_dma_->start();
        pio_sm_set_enabled(engine_->block, engine_->node, true);
        prefix_dma_->start();
        return true;
    }

    bool Device::begin_read(std::span<const uint8_t> prefix, std::span<uint8_t> payload, Completion completion) {
        std::lock_guard<Mutex> lock(state_mutex_);

        if (!is_loaded() || busy_.load() || prefix.empty() || payload.empty()) { return false; }

        if (!prepare_read(prefix.size(), payload.size()) || !configure_rx(payload) || !configure_tx(*prefix_dma_, prefix)) {
            quiesce();
            return false;
        }

        completion_ = std::move(completion);
        operation_  = Operation::Read;
        busy_.store(true);

        // RX is armed first so the PIO FIFO cannot overrun at the read
        // turn-around.
        rx_dma_->start();
        pio_sm_set_enabled(engine_->block, engine_->node, true);
        prefix_dma_->start();
        return true;
    }

    void Device::complete(bool ok) {
        Completion callback;
        {
            std::lock_guard<Mutex> lock(state_mutex_);
            if (!busy_.load()) { return; }

            pio_sm_set_enabled(engine_->block, engine_->node, false);
            pio_sm_exec(engine_->block, engine_->node, pio_encode_mov(pio_pins, pio_null));

            callback   = std::move(completion_);
            operation_ = Operation::None;
            busy_.store(false);
        }

        if (callback) { callback(ok); }
    }

    bool Device::write(std::span<const uint8_t> prefix, std::span<const uint8_t> payload) {
        Semaphore done{0, 1};
        bool      result = false;

        if (!begin_write(prefix, payload, [&](bool ok) {
                result = ok;
                done.release();
            })) {
            return false;
        }

        done.acquire();
        return result;
    }

    bool Device::read(std::span<const uint8_t> prefix, std::span<uint8_t> payload) {
        Semaphore done{0, 1};
        bool      result = false;

        if (!begin_read(prefix, payload, [&](bool ok) {
                result = ok;
                done.release();
            })) {
            return false;
        }

        done.acquire();
        return result;
    }

} // namespace corekit::Qspi
