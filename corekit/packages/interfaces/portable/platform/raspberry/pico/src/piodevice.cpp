#include "corekit/piodevice.hpp"

#include <hardware/pio.h>

#include <cstdint>
#include <format>
#include <string>

#include "corekit/check.hpp"
#include "corekit/gpiodevice.hpp"

bool pio_sm_is_enabled(PIO block, uint sm) {
    check_pio_param(block);
    check_sm_param(sm);
    return (block->ctrl & (1u << sm)) != 0;
}

namespace corekit::Pio {

    Program::Program(const pio_program_t& program)
        : pio_program_t(program) { }

    Program::State& Program::requestState(PIO block) const {
        if (!states.contains(block)) { states[block] = State(); }

        return states.at(block);
    }

    bool Program::isInstalled(PIO block) const {
        const State& state = requestState(block);
        return state.adress.has_value();
    }

    int Program::install(PIO block, uint node) {
        State& state = requestState(block);

        if (!isInstalled(block)) {
            corecheck(pio_can_add_program(block, this), RuntimeError("Cannot install PIO program: not enough space left"));

            state.adress   = pio_add_program(block, this);
            state.modified = false;
            state.nodemask = 0;
        }

        state.nodemask |= 1u << node;

        return state.adress.value();
    }

    void Program::uninstall(PIO block, uint node) {
        State& state    = requestState(block);

        state.nodemask &= ~(1u << node);

        if (isInstalled(block)) {
            const bool is_unused = state.nodemask == 0;

            if (is_unused) {
                pio_remove_program(block, this, state.adress.value());

                state.adress.reset();
                state.modified = false;
            }
        }
    }

    bool Program::modify(PIO block, uint line, Command command) {
        corecheck(isInstalled(block), RuntimeError("Cannot modify a PIO program that is not installed."));

        State& state    = requestState(block);

        const uint base = state.adress.value();

        corecheck(line < length, OutOfRangeError(std::format("Cannot modify a PIO program line that is out of range: " "line={} length={}", line, length)));

        block->instr_mem[base + line] = command;

        return true;
    }

    // --------------------------------------------------------------
    // Node Implementation
    // --------------------------------------------------------------

    Node::Node(const PIO block, Program::Ptr program)
        : Node(block, pio_claim_unused_sm(block, true), std::move(program)) { }

    Node::Node(const PIO block, uint node, Program::Ptr program)
        : AsyncDevice<uint32_t>(std::format("PIO{}-{}", pio_get_index(block), node), {&block->txf[node], pio_get_dreq(block, node, true)}, {&block->rxf[node], pio_get_dreq(block, node, false)})
        , block(block)
        , node(node)
        , program(std::move(program)) { }

    Node::~Node() {
        pio_sm_set_enabled(block, node, false);
        pio_sm_restart(block, node);

        pio_sm_unclaim(block, node);
    }

    uint Node::unique_id() const { return pio_get_index(block) * NUM_PIO_STATE_MACHINES + node; }

    bool Node::is_running() const { return pio_sm_is_enabled(block, node); }

    // --------------------------------------------------------------
    // Loading
    // --------------------------------------------------------------

    bool Node::on_load() {
        corecheck(program != nullptr, RuntimeError("Cannot load a PIO node without a program."));

        const int base = program->install(block, node);

        corecheck(base >= 0, RuntimeError("Failed to install PIO program for node: " + name));

        NodeConf node_cfg = pio_get_default_sm_config();

        corecheck(build_node_conf(node_cfg, base), RuntimeError("Failed to build PIO node configuration."));

        LaunchConf launch_cfg;

        corecheck(build_launch_conf(launch_cfg), RuntimeError("Failed to build PIO launch configuration."));

        const uint initial_pc = base + launch_cfg.entrypoint;

        corecheck(pio_sm_init(block, node, initial_pc, &node_cfg) == PICO_OK, RuntimeError("Failed to initialize PIO state machine."));

        if (configure_pins(launch_cfg.output_pins, true)) {
            const Gpio::Range& pins = launch_cfg.output_pins.value();

            pio_sm_set_out_pins(block, node, pins.lower(), pins.count());
        }

        if (configure_pins(launch_cfg.input_pins, false)) {
            const Gpio::Range& pins = launch_cfg.input_pins.value();

            pio_sm_set_in_pins(block, node, pins.lower());
        }

        if (configure_pins(launch_cfg.set_pins, true)) {
            const Gpio::Range& pins = launch_cfg.set_pins.value();

            pio_sm_set_set_pins(block, node, pins.lower(), pins.count());
        }

        if (configure_pins(launch_cfg.side_pins, true)) {
            const Gpio::Range& pins = launch_cfg.side_pins.value();

            pio_sm_set_sideset_pins(block, node, pins.lower());
        }

        if (configure_pins(launch_cfg.jump_pin, false)) {
            const Gpio::Range& pins = launch_cfg.jump_pin.value();

            pio_sm_set_jmp_pin(block, node, pins.lower());
        }

        corecheck(configure_regs(pio_x, launch_cfg.scratchX), RuntimeError("Failed to preload PIO scratch X register."));

        corecheck(configure_regs(pio_y, launch_cfg.scratchY), RuntimeError("Failed to preload PIO scratch Y register."));

        corecheck(configure_regs(pio_isr, launch_cfg.isr), RuntimeError("Failed to preload PIO input shift register."));

        corecheck(configure_regs(pio_osr, launch_cfg.osr), RuntimeError("Failed to preload PIO output shift register."));

        corecheck(configure_dmas(), RuntimeError("Failed to configure PIO DMAs."));

        pio_sm_set_enabled(block, node, launch_cfg.autostart);

        return true;
    }

    bool Node::on_unload() {
        pio_sm_set_enabled(block, node, false);
        pio_sm_restart(block, node);

        if (program == nullptr) { return false; }

        program->uninstall(block, node);

        return true;
    }

    // --------------------------------------------------------------
    // Configuration
    // --------------------------------------------------------------

    bool Node::build_launch_conf(LaunchConf& launchConf) { return true; }

    bool Node::build_node_conf(NodeConf& nodeConf, uint base) { return true; }

    bool Node::configure_regs(pio_src_dest reg, const PreloadVal& val) {
        static const Command pull_cmd = pio_encode_pull(false, false);

        const Command mov_cmd         = pio_encode_mov(reg, pio_osr);

        if (!val.has_value()) { return true; }

        corecheck(!is_running(), RuntimeError("Cannot preload a PIO register while the state machine " "is running."));

        pio_sm_put(block, node, val.value());
        pio_sm_exec(block, node, pull_cmd);
        pio_sm_exec(block, node, mov_cmd);

        return true;
    }

    bool Node::configure_pins(const PinoutCfg& config, bool is_output) {
        if (!config.has_value()) { return false; }

        const Gpio::Range& pins = config.value();

        for (const Gpio::Pin pin : pins.pins()) { pio_gpio_init(block, pin); }

        uint base = pio_get_gpio_base(block);

        corecheck(base <= pins.lower(), RuntimeError(std::format("Cannot configure PIO pins: GPIO base is too high: " "base={} pin={}", base, pins.lower())));

        corecheck(pins.upper() <= base + 32, RuntimeError(std::format("Cannot configure PIO pins: GPIO base is too low: " "base={} pin={}", base, pins.upper())));

        pio_sm_set_consecutive_pindirs(block, node, pins.lower(), pins.count(), is_output);

        return true;
    }

    // --------------------------------------------------------------
    // DMA
    // --------------------------------------------------------------

    bool Node::configure_dmas() { return true; }

    // --------------------------------------------------------------
    // IO
    // --------------------------------------------------------------

    bool Node::write(const uint32_t& data) {
        pio_sm_put_blocking(block, node, data);
        return true;
    }

    bool Node::write_burst(std::span<const uint32_t> data) {
        for (const uint32_t& value : data) {
            if (!write(value)) { return false; }
        }

        return true;
    }

    bool Node::read(uint32_t& data) {
        data = pio_sm_get_blocking(block, node);
        return true;
    }

    bool Node::read_burst(std::span<uint32_t> data) {
        for (uint32_t& value : data) {
            if (!read(value)) { return false; }
        }

        return true;
    }

} // namespace corekit::Pio