#pragma once

#include <hardware/pio.h>

#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "corekit/asyncdevice.hpp"
#include "corekit/result.hpp"

extern bool pio_sm_is_enabled(PIO block, uint sm);

namespace corekit::Pio {

    using Command  = uint16_t;
    using NodeConf = pio_sm_config;
    using Address  = std::optional<uint>;

    struct LaunchConf {
        using PreloadVal = std::optional<uint32_t>;

        bool       autostart  = true;
        uint       entrypoint = 0;
        PreloadVal scratchX   = std::nullopt;
        PreloadVal scratchY   = std::nullopt;
        PreloadVal isr        = std::nullopt;
        PreloadVal osr        = std::nullopt;
    };

    class Node;

    struct Program : public pio_program {
        friend class Node;

        struct State {
            State() {
                reset();
            }

            void reset() {
                adress.reset();
                modified = false;
                nodemask = 0;
            }

            Address adress;
            bool    modified;
            uint8_t nodemask;
        };

       public:
        using Ptr    = std::shared_ptr<Program>;
        using Target = std::shared_ptr<Node>;
        Program(const pio_program_t& program);

        virtual bool install(PIO block) final;
        virtual bool uninstall(PIO block) final;
        virtual bool isInstalled(PIO block) const final;
        virtual bool modify(PIO block, uint line, Command command) final;

        virtual NodeConf   buildNodeConf(PIO block, uint node, uint base);
        virtual LaunchConf buildLaunchConf(PIO block, uint node);

        virtual VoidResult configurePins(Target instance);
        virtual VoidResult configureDmas(Target instance);

        virtual const State& getState(PIO block) const final;

       private:
        virtual bool registerNode(PIO block, uint node) final;
        virtual bool unregisterNode(PIO block, uint node) final;

        virtual State& requestState(PIO block) const final;

        mutable std::map<PIO, State> states;
    };

    class Node
        : public AsyncDevice<uint32_t>
        , public std::enable_shared_from_this<Node> {
        friend class Program;

       public:
        using Ptr  = std::shared_ptr<Node>;
        using List = std::vector<Ptr>;

        Node(const PIO block, uint node);
        virtual ~Node() override;

        static Ptr request_unused(const PIO block);

        bool deploy(const Program::Ptr& program);
        bool isRunning() const;
        uint unique_id() const;

        virtual bool write(const uint32_t& data) override;
        virtual bool write_bulk(std::span<const uint32_t> data) override;
        virtual bool read(uint32_t& data) override;
        virtual bool read_bulk(std::span<uint32_t> data) override;

        const PIO  block;
        const uint node;

       protected:
        virtual bool on_load() override;
        virtual bool on_unload() override;

        void preloadReg(pio_src_dest reg, uint32_t value);

        Program::Ptr program;
    };

}  // namespace corekit::Pio