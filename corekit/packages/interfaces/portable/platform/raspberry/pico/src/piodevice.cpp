#include "corekit/piodevice.hpp"

#include <hardware/pio.h>

#include <cstdint>
#include <format>
#include <string>

#include "corekit/gpiodevice.hpp"

bool pio_sm_is_enabled(PIO block, uint sm) {
    check_pio_param(block);
    check_sm_param(sm);
    return (block->ctrl & (1u << sm)) != 0;
}

namespace corekit::Pio {

    // --------------------------------------------------------------
    // Program Implementation
    // --------------------------------------------------------------

    Program::Program(const pio_program_t& program) : pio_program_t(program) {}

    Program::State& Program::requestState(PIO block) const {
        if (!states.contains(block)) {
            states[block] = State();
        }

        return states.at(block);
    }

    bool Program::isInstalled(PIO block) const {
        const State& state = requestState(block);
        return state.adress.has_value();
    }

    int Program::install(PIO block, uint node) {
        State& state = requestState(block);

        if (!isInstalled(block)) {
            if (!pio_can_add_program(block, this)) {
                Error::stack.push(RuntimeError(
                    "Cannot install PIO program: not enough space left"));

                return -1;
            }

            state.adress   = pio_add_program(block, this);
            state.modified = false;
            state.nodemask = 0;
        }

        state.nodemask |= 1 << node;
        return state.adress.value();
    }

    void Program::uninstall(PIO block, uint node) {
        State& state = requestState(block);
        state.nodemask &= ~(1 << node);

        if (isInstalled(block)) {
            const bool is_unused = (state.nodemask == 0);

            if (is_unused) {
                pio_remove_program(block, this, state.adress.value());
                state.adress.reset();
                state.modified = false;
            }
        }
    }

    bool Program::modify(PIO block, uint line, Command command) {
        if (!isInstalled(block)) {
            Error::stack.push(RuntimeError(
                "Cannot modify a PIO program that is not installed."));
            return false;
        }

        State& state = requestState(block);

        const uint base = state.adress.value();

        if (length <= line) {
            Error::stack.push(OutOfRangeError(
                std::format("Cannot modify a PIO program line that is out of "
                            "range: line={} length={}",
                            line,
                            length)));
            return false;
        }

        block->instr_mem[base + line] = command;
        return true;
    }

    // --------------------------------------------------------------
    // Node<T> Implementation
    // --------------------------------------------------------------

    Node::Node(const PIO block, Program::Ptr program)
        : Node::Node(block, pio_claim_unused_sm(block, true), program) {}

    Node::Node(const PIO block, uint node, Program::Ptr program)
        : AsyncDevice<uint32_t>(
              std::format("PIO{}-{}", pio_get_index(block), node),
              {&block->txf[node], pio_get_dreq(block, node, true)},  //
              {&block->rxf[node], pio_get_dreq(block, node, false)}  //
              )
        , block(block)
        , node(node)
        , program(program) {}

    Node::~Node() {
        pio_sm_set_enabled(block, node, false);
        pio_sm_restart(block, node);
        pio_sm_unclaim(block, node);
    }

    uint Node::unique_id() const {
        return pio_get_index(block) * NUM_PIO_STATE_MACHINES + node;
    }

    bool Node::is_running() const {
        return pio_sm_is_enabled(block, node);
    }

    bool Node::on_load() {
        if (!program) {
            Error::stack.push(RuntimeError(
                "Cannot load a PIO node without a program. Use deploy() to "
                "assign a program to the node first."));
            return false;
        }

        const int base = program->install(block, node);
        if (base < 0) {
            Error::stack.push(RuntimeError(
                "Failed to install PIO program for node: " + name));
            return false;
        }

        NodeConf node_cfg = pio_get_default_sm_config();
        if (!build_node_conf(node_cfg, base)) {
            Error::stack.push(
                RuntimeError("Failed to build PIO node configuration."));
            return false;
        }

        LaunchConf launch_cfg;
        if (!build_launch_conf(launch_cfg)) {
            Error::stack.push(
                RuntimeError("Failed to build PIO launch configuration."));
            return false;
        }

        const uint initial_pc = base + launch_cfg.entrypoint;
        if (pio_sm_init(block, node, initial_pc, &node_cfg) != PICO_OK) {
            Error::stack.push(
                RuntimeError("Failed to initialize PIO state machine."));
            return false;
        }

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

        if (!configure_regs(pio_x, launch_cfg.scratchX)) {
            Error::stack.push(
                RuntimeError("Failed to preload PIO scratch X register."));
            return false;
        }

        if (!configure_regs(pio_y, launch_cfg.scratchY)) {
            Error::stack.push(
                RuntimeError("Failed to preload PIO scratch Y register."));
            return false;
        }

        if (!configure_regs(pio_isr, launch_cfg.isr)) {
            Error::stack.push(
                RuntimeError("Failed to preload PIO ISR register."));
            return false;
        }

        if (!configure_regs(pio_osr, launch_cfg.osr)) {
            Error::stack.push(
                RuntimeError("Failed to preload PIO OSR register."));
            return false;
        }

        if (!configure_dmas()) {
            Error::stack.push(RuntimeError("Failed to configure PIO DMAs."));
            return false;
        }

        pio_sm_set_enabled(block, node, launch_cfg.autostart);
        return true;
    }

    bool Node::on_unload() {
        pio_sm_set_enabled(block, node, false);
        pio_sm_restart(block, node);

        if (program == nullptr) {
            return false;
        }

        program->uninstall(block, node);

        return true;
    }

    bool Node::build_launch_conf(LaunchConf& launchConf) {
        return true;
    }

    bool Node::build_node_conf(NodeConf& nodeConf, uint base) {
        return true;
    }

    bool Node::configure_regs(pio_src_dest reg, const PreloadVal& val) {
        static const Command pullCmd = pio_encode_pull(false, false);
        const Command        movCmd  = pio_encode_mov(reg, pio_osr);

        if (!val.has_value()) {
            return true;
        }

        if (is_running()) {
            Error::stack.push(RuntimeError(
                "Cannot preload a PIO register while the state machine is "
                "running."));
            return false;
        }

        pio_sm_put(block, node, val.value());
        pio_sm_exec(block, node, pullCmd);
        pio_sm_exec(block, node, movCmd);
        return true;
    }

    bool Node::configure_pins(const PinoutCfg& config, bool is_output) {
        if (config.has_value()) {
            const Gpio::Range pins = config.value();

            for (const Gpio::Pin pin : pins.pins()) {
                pio_gpio_init(block, pin);
            }

            pio_sm_set_consecutive_pindirs(block,
                                           node,
                                           pins.lower(),
                                           pins.count(),
                                           is_output);
            return true;
        }

        return false;
    };

    bool Node::configure_dmas() {
        return true;
    }

    bool Node::write(const uint32_t& data) {
        pio_sm_put_blocking(block, node, data);
        return true;
    }

    bool Node::write_burst(std::span<const uint32_t> data) {
        for (const uint32_t& value : data) {
            if (!write(value))
                return false;
        }

        return true;
    }

    bool Node::read(uint32_t& data) {
        data = pio_sm_get_blocking(block, node);
        return true;
    }

    bool Node::read_burst(std::span<uint32_t> data) {
        for (uint32_t& value : data) {
            if (!read(value))
                return false;
        }

        return true;
    }

}  // namespace corekit::Pio