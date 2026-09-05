#include "corekit/piodevice.hpp"

#include <hardware/pio.h>

#include <cstdint>
#include <string>

#include "corekit/error.hpp"
#include "corekit/logger.hpp"
#include "corekit/result.hpp"

bool pio_sm_is_enabled(PIO block, uint sm) {
    check_pio_param(block);
    check_sm_param(sm);
    return (block->ctrl & bool_to_bit(1 << sm)) != 0;
}

namespace corekit::Pio {

    Mutex Pio::Node::claim_mutex;

    const Logger logger("PIO");

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

    VoidResult Program::install(PIO block) {
        if (!isInstalled(block)) {
            State& state = requestState(block);

            if (!pio_can_add_program(block, this)) {
                return RuntimeError(
                    "Cannot install a PIO program that does not fit in the "
                    "PIO instruction memory.");
            }

            state.adress   = pio_add_program(block, this);
            state.modified = false;
            state.nodemask = 0;
            return VoidResult();
        }

        return VoidResult();
    }

    VoidResult Program::uninstall(PIO block) {
        if (isInstalled(block)) {
            State& state = requestState(block);

            if (state.nodemask != 0) {
                return RuntimeError(
                    "Cannot uninstall a PIO program that has registered "
                    "nodes.");
            }

            pio_remove_program(block, this, state.adress.value());
            state.reset();
        }

        return VoidResult();
    }

    VoidResult Program::modify(PIO block, uint line, Command command) {
        if (!isInstalled(block)) {
            return RuntimeError(
                "Cannot modify a PIO program that is not installed.");
        }

        State& state = requestState(block);

        const uint base = state.adress.value_or(0);

        if (length <= line) {
            return OutOfRangeError(
                "Cannot modify a PIO program line that is out of code "
                "range.");
        }

        block->instr_mem[base + line] = command;
        return VoidResult();
    }

    VoidResult Program::registerNode(PIO block, uint node) {
        if (!isInstalled(block)) {
            return RuntimeError(
                "Cannot register a node to a PIO program that is not "
                "installed.");
        }

        State& state = requestState(block);
        state.nodemask |= (1 << node);
        return VoidResult();
    }

    void Program::unregisterNode(PIO block, uint node) {
        if (!isInstalled(block)) {
            return;
        }

        State& state = requestState(block);
        state.nodemask &= ~(1 << node);
    }

    LaunchConf Program::buildLaunchConf(PIO block, uint node) {
        return LaunchConf();
    }

    NodeConf Program::buildNodeConf(PIO block, uint node, uint base) {
        return pio_get_default_sm_config();
    }

    VoidResult Program::configurePins(PIO block, uint node) {
        return VoidResult();
    }

    VoidResult Program::configureDmas(PIO                block,
                                      uint               node,
                                      CtrlBlock          writer,
                                      CtrlBlock          reader,
                                      Dma::Device::List& dmas) {
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

    VoidResult Node::deploy(const Program::Ptr& program) {
        logger() << std::format("deploying program to {}", name, node);
        if (this->program != program && this->is_loaded()) {
            logger() << std::format("do unload");
            this->unload();
        }

        logger() << std::format("stage 0: initializing");
        this->program = program;
        return this->load();
    }

    bool Node::isRunning() const {
        return pio_sm_is_enabled(block, node);
    }

    VoidResult Node::on_load() {
        logger() << std::format("loading program for {}", name, node);
        if (!program) {
            return RuntimeError(
                "Cannot load a PIO node without a program. Use deploy() to "
                "assign a program to the node first.");
        }

        const Program::State& state = program->getState(block);

        if (state.nodemask == 0) {
            if (!program->install(block)) {
                return RuntimeError("Failed to install PIO program on PIO.");
            }
        }

        if (!program->registerNode(block, node)) {
            return RuntimeError(
                "Failed to register PIO node with PIO program.");
        }

        const uint       base = state.adress.value_or(0);
        const NodeConf   ncfg = program->buildNodeConf(block, node, base);
        const LaunchConf lcfg = program->buildLaunchConf(block, node);

        if (!program->configurePins(block, node)) {
            return RuntimeError("Failed to configure PIO pins.");
        }

        const uint initial_pc = base + lcfg.entrypoint;
        const uint result     = pio_sm_init(block, node, initial_pc, &ncfg);

        if (result != PICO_OK) {
            return RuntimeError("Failed to initialize PIO state machine.");
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

        dmalist.clear();
        if (!program->configureDmas(block, node, writer, reader, dmalist)) {
            return RuntimeError("Failed to configure PIO DMAs.");
        }

        pio_sm_set_enabled(block, node, lcfg.autostart);

        return VoidResult();
    }

    VoidResult Node::on_unload() {
        pio_sm_set_enabled(block, node, false);
        pio_sm_restart(block, node);
        dmalist.clear();

        if (program == nullptr) {
            return RuntimeError(
                "Cannot unregister a node from a null program.");
        }

        program->unregisterNode(block, node);
        return VoidResult();
    }

    VoidResult Node::preloadReg(pio_src_dest reg, uint32_t value) {
        static const Command pullCmd = pio_encode_pull(false, false);
        static const Command movCmd  = pio_encode_mov(reg, pio_osr);

        if (isRunning()) {
            return RuntimeError(
                "Cannot preload a PIO register while the state machine is "
                "running.");
        }

        pio_sm_put(block, node, value);
        pio_sm_exec(block, node, pullCmd);
        pio_sm_exec(block, node, movCmd);
        return VoidResult();
    }

    VoidResult Node::write(const uint32_t& data) {
        pio_sm_put_blocking(block, node, data);
        return VoidResult();
    }

    VoidResult Node::write_burst(std::span<const uint32_t> data) {
        for (const uint32_t& value : data) {
            if (!write(value))
                return RuntimeError(
                    "Failed to write burst of data to PIO state machine.");
        }

        return VoidResult();
    }

    VoidResult Node::read(uint32_t& data) {
        data = pio_sm_get_blocking(block, node);
        return VoidResult();
    }

    VoidResult Node::read_burst(std::span<uint32_t> data) {
        for (uint32_t& value : data) {
            if (!read(value))
                return RuntimeError(
                    "Failed to read burst of data from PIO state machine.");
        }

        return VoidResult();
    }

}  // namespace corekit::Pio