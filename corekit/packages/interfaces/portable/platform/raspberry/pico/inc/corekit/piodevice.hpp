#pragma once

#include <hardware/pio.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "corekit/asyncdevice.hpp"
#include "corekit/dmadevice.hpp"
#include "corekit/gpiodevice.hpp"
#include "corekit/result.hpp"

extern bool pio_sm_is_enabled(PIO block, uint sm);

namespace corekit::Pio {

    using Command    = uint16_t;
    using NodeConf   = pio_sm_config;
    using Address    = std::optional<uint>;
    using PreloadVal = std::optional<uint32_t>;
    using PinoutCfg  = std::optional<Gpio::Range>;

    struct LaunchConf {
        bool autostart  = true;
        uint entrypoint = 0;

        PreloadVal scratchX = std::nullopt;
        PreloadVal scratchY = std::nullopt;
        PreloadVal isr      = std::nullopt;
        PreloadVal osr      = std::nullopt;

        PinoutCfg output_pins = std::nullopt;
        PinoutCfg input_pins  = std::nullopt;
        PinoutCfg set_pins    = std::nullopt;
        PinoutCfg side_pins   = std::nullopt;
    };

    struct Program : public pio_program {
        friend class Node;

        struct State {
            Address adress   = std::nullopt;
            bool    modified = false;
            uint8_t nodemask = 0;
        };

       public:
        using Ptr = std::shared_ptr<Program>;
        Program(const pio_program_t& program);

        int  install(PIO block, uint node);
        void uninstall(PIO block, uint node);
        bool isInstalled(PIO block) const;
        bool modify(PIO block, uint line, Command command);

       private:
        State& requestState(PIO block) const;

        mutable std::map<PIO, State> states;
    };

    class Node : public AsyncDevice<uint32_t> {
        friend class Program;

       public:
        using Ptr  = std::shared_ptr<Node>;
        using List = std::vector<Ptr>;

        Node(const PIO block, Program::Ptr program);
        virtual ~Node() override;

        bool is_running() const;
        uint unique_id() const;

        virtual bool write(const uint32_t& data) override final;
        virtual bool write_burst(std::span<const uint32_t> data) override final;
        virtual bool read(uint32_t& data) override final;
        virtual bool read_burst(std::span<uint32_t> data) override final;

        const PIO  block;
        const uint node;

        const Program::Ptr program;

       protected:
        virtual bool build_launch_conf(LaunchConf& launch_cfg);
        virtual bool build_node_conf(NodeConf& node_cfg, uint base);
        virtual bool configure_dmas();

       private:
        Node(const PIO block, uint node, Program::Ptr program);

        virtual bool on_load() override final;
        virtual bool on_unload() override final;

        bool configure_regs(pio_src_dest reg, const PreloadVal& val);
        bool configure_pins(const PinoutCfg& pins, bool is_output);
    };

}  // namespace corekit::Pio