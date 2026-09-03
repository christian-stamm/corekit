#include "corekit/piodevice.hpp"

#include <hardware/pio.h>

#include <cstdint>
#include <string>

bool pio_sm_is_enabled(PIO block, uint sm) {
    check_pio_param(block);
    check_sm_param(sm);
    return (block->ctrl & bool_to_bit(1 << sm)) != 0;
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

    const Program::State& Program::getState(PIO block) const {
        return requestState(block);
    }

    bool Program::isInstalled(PIO block) const {
        State& state = requestState(block);
        return state.adress.has_value();
    }

    bool Program::install(PIO block) {
        if (!isInstalled(block)) {
            State& state = requestState(block);

            if (pio_can_add_program(block, this)) {
                state.adress   = pio_add_program(block, this);
                state.modified = false;
                state.nodemask = 0;
                return true;
            }
        }

        return false;
    }

    bool Program::uninstall(PIO block) {
        if (isInstalled(block)) {
            State& state = requestState(block);

            if (state.nodemask != 0) {
                throw std::runtime_error(
                    "Cannot uninstall a PIO program that has registered "
                    "nodes.");
            }

            pio_remove_program(block, this, state.adress.value());
            state.reset();
            return true;
        }

        return false;
    }

    bool Program::modify(PIO block, uint line, Command command) {
        if (!isInstalled(block)) {
            throw std::runtime_error(
                "Cannot modify a PIO program that is not installed.");
        }

        State& state = requestState(block);

        const uint base = state.adress.value_or(0);

        if (length <= line) {
            throw std::runtime_error(
                "Cannot modify a PIO program line that is out of code "
                "range.");
        }

        block->instr_mem[base + line] = command;
        return true;
    }

    bool Program::registerNode(PIO block, uint node) {
        if (isInstalled(block) == false) {
            throw std::runtime_error(
                "Cannot register a node to a PIO program that is not "
                "installed.");
        }

        State&        state = requestState(block);
        const uint8_t mask  = (1 << node);

        if (state.nodemask & mask) {
            return false;
        }

        state.nodemask |= mask;
        return true;
    }

    bool Program::unregisterNode(PIO block, uint node) {
        if (isInstalled(block) == false) {
            throw std::runtime_error(
                "Cannot unregister a node from a PIO program that is not "
                "installed.");
        }

        State&     state = requestState(block);
        const uint mask  = (1 << node);

        if ((state.nodemask & mask) == 0) {
            return false;
        }

        state.nodemask &= ~mask;
        return true;
    }

    LaunchConf Program::buildLaunchConf(PIO block, uint node) {
        return LaunchConf();
    }

    NodeConf Program::buildNodeConf(PIO block, uint node, uint base) {
        return pio_get_default_sm_config();
    }

    VoidResult Program::configurePins(Program::Target instance) {
        return VoidResult();
    }

    VoidResult Program::configureDmas(Program::Target instance) {
        return VoidResult();
    }

    // --------------------------------------------------------------
    // Node<T> Implementation
    // --------------------------------------------------------------

    Node::Node(const PIO block, uint node)
        : AsyncDevice<uint32_t>(
              std::format("PIO{}-{}", pio_get_index(block), node),
              {&block->txf[node], pio_get_dreq(block, node, true)},  //
              {&block->rxf[node], pio_get_dreq(block, node, false)}  //
              )
        , block(block)
        , node(node) {
        pio_sm_claim(block, node);
    }

    Node::~Node() {
        pio_sm_unclaim(block, node);
    }

    uint Node::unique_id() const {
        return pio_get_index(block) * NUM_PIO_STATE_MACHINES + node;
    }

    bool Node::deploy(const Program::Ptr& program) {
        if (this->program != program) {
            this->unload();
        }

        this->program = program;
        return this->load();
    }

    bool Node::isRunning() const {
        return pio_sm_is_enabled(block, node);
    }

    bool Node::on_load() {
        if (!program) {
            throw std::runtime_error(
                "Cannot load a PIO node without a program. Use deploy() to "
                "assign a program to the node first.");
        }

        const Program::State& state = program->getState(block);

        if (state.nodemask == 0) {
            if (!program->install(block)) {
                throw std::runtime_error(
                    "Failed to install PIO program on PIO.");
            }
        }

        if (program->registerNode(block, node)) {
            const uint       base = state.adress.value_or(0);
            const NodeConf   ncfg = program->buildNodeConf(block, node, base);
            const LaunchConf lcfg = program->buildLaunchConf(block, node);

            if (!program->configurePins(shared_from_this())) {
                return false;
            }

            const uint initial_pc = base + lcfg.entrypoint;
            const uint result     = pio_sm_init(block, node, initial_pc, &ncfg);

            if (result != PICO_OK) {
                throw std::runtime_error(
                    "Failed to initialize PIO state machine.");
            }

            if (lcfg.scratchX.has_value()) {
                preloadReg(pio_x, lcfg.scratchX.value());
            }

            if (lcfg.scratchY.has_value()) {
                preloadReg(pio_y, lcfg.scratchY.value());
            }

            if (lcfg.isr.has_value()) {
                preloadReg(pio_isr, lcfg.isr.value());
            }

            if (lcfg.osr.has_value()) {
                preloadReg(pio_osr, lcfg.osr.value());
            }

            if (!program->configureDmas(shared_from_this())) {
                return false;
            }

            pio_sm_set_enabled(block, node, lcfg.autostart);

            return true;
        }

        return false;
    }

    bool Node::on_unload() {
        pio_sm_set_enabled(block, node, false);
        pio_sm_restart(block, node);

        if (program == nullptr) {
            return false;
        }

        return program->unregisterNode(block, node);
    }

    void Node::preloadReg(pio_src_dest reg, uint32_t value) {
        static const Command pullCmd = pio_encode_pull(false, false);
        static const Command movCmd  = pio_encode_mov(reg, pio_osr);

        if (isRunning()) {
            throw std::runtime_error(
                "Cannot preload a PIO register while the state machine is "
                "running.");
        }

        pio_sm_put(block, node, value);
        pio_sm_exec(block, node, pullCmd);
        pio_sm_exec(block, node, movCmd);
    }

    bool Node::write(const uint32_t& data) {
        pio_sm_put_blocking(block, node, data);
        return true;
    }

    bool Node::write_bulk(std::span<const uint32_t> data) {
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

    bool Node::read_bulk(std::span<uint32_t> data) {
        for (uint32_t& value : data) {
            if (!read(value))
                return false;
        }

        return true;
    }

}  // namespace corekit::Pio