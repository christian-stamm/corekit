#include "corekit/platform/piodevice.hpp"

#include <hardware/gpio.h>
#include <hardware/pio.h>

#include <cstdint>
#include <string>

bool pio_sm_is_enabled(PIO block, uint sm) {
    check_pio_param(block);
    check_sm_param(sm);
    return (block->ctrl & bool_to_bit(1 << sm)) != 0;
}

namespace corekit::platform::pio {

    template class Node<uint8_t>;
    template class Node<uint16_t>;
    template class Node<uint32_t>;

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

    // --------------------------------------------------------------
    // Node<T> Implementation
    // --------------------------------------------------------------

    template <typename T>
    Node<T>::Node(const PIO block, uint node)
        : AsyncDevice<T>(
              std::format("PIO{}-{}", pio_get_index(block), node),
              {&block->txf[node], pio_get_dreq(block, node, true)},  //
              {&block->rxf[node], pio_get_dreq(block, node, false)}  //
              )
        , block(block)
        , node(node) {
        pio_sm_claim(block, node);
    }

    template <typename T>
    Node<T>::~Node() {
        pio_sm_unclaim(block, node);
    }

    template <typename T>
    Node<T>::Ptr Node<T>::requestUnused(PIO block) {
        const int node = pio_claim_unused_sm(block, false);

        if (node < 0) {
            throw std::runtime_error(
                "Failed to claim a free PIO state machine.");
        }

        pio_sm_unclaim(block, node);
        return std::make_shared<Node<T>>(block, (uint)(node));
    }

    template <typename T>
    bool Node<T>::deploy(const Program::Ptr& program) {
        if (this->program != program) {
            this->unload();
        }

        this->program = program;
        return this->load();
    }

    template <typename T>
    bool Node<T>::isRunning() const {
        return pio_sm_is_enabled(block, node);
    }

    template <typename T>
    bool Node<T>::on_load() {
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
            const uint       initial_pc = base + lcfg.entrypoint;

            const uint result = pio_sm_init(block, node, initial_pc, &ncfg);

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

            pio_sm_set_enabled(block, node, lcfg.autostart);

            return true;
        }

        return false;
    }

    template <typename T>
    bool Node<T>::on_unload() {
        pio_sm_set_enabled(block, node, false);
        pio_sm_restart(block, node);

        if (program == nullptr) {
            return false;
        }

        return program->unregisterNode(block, node);
    }

    template <typename T>
    void Node<T>::preloadReg(pio_src_dest reg, uint32_t value) {
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

    template <typename T>
    bool Node<T>::write(const T& data) {
        pio_sm_put_blocking(block, node, data);
        return true;
    }

    template <typename T>
    bool Node<T>::write_bulk(std::span<const T> data) {
        for (const T& value : data) {
            if (!write(value))
                return false;
        }

        return true;
    }

    template <typename T>
    bool Node<T>::read(T& data) {
        data = pio_sm_get_blocking(block, node);
        return true;
    }

    template <typename T>
    bool Node<T>::read_bulk(std::span<T> data) {
        for (T& value : data) {
            if (!read(value))
                return false;
        }

        return true;
    }

}  // namespace corekit::platform::pio