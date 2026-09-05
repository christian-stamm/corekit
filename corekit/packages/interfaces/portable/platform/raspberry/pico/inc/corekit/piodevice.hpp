#pragma once

#include <hardware/pio.h>

#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "corekit/asyncdevice.hpp"
#include "corekit/dmadevice.hpp"
#include "corekit/mutex.hpp"
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
        using Ptr = std::shared_ptr<Program>;

        Program(const pio_program_t& program);

        virtual VoidResult install(PIO block) final;
        virtual VoidResult uninstall(PIO block) final;
        virtual bool       isInstalled(PIO block) const final;
        virtual VoidResult modify(PIO block, uint line, Command command) final;

        virtual NodeConf   buildNodeConf(PIO block, uint node, uint base);
        virtual LaunchConf buildLaunchConf(PIO block, uint node);

        virtual VoidResult configurePins(PIO block, uint node);
        virtual VoidResult configureDmas(PIO                block,
                                         uint               node,
                                         CtrlBlock          writer,
                                         CtrlBlock          reader,
                                         Dma::Device::List& dmas);

        virtual const State& getState(PIO block) const final;

       private:
        virtual VoidResult registerNode(PIO block, uint node) final;
        virtual void       unregisterNode(PIO block, uint node) final;

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

        template <typename T = Node>
        static std::shared_ptr<T> request_unused(const PIO block) {
            std::lock_guard lock(claim_mutex);
            const int       node = pio_claim_unused_sm(block, false);

            if (node < 0) {
                return nullptr;
            }

            pio_sm_unclaim(block, node);
            return std::make_shared<T>(block, (uint)(node));
        }

        VoidResult deploy(const Program::Ptr& program);

        bool isRunning() const;
        uint unique_id() const;

        virtual VoidResult write(const uint32_t& data) override;
        virtual VoidResult write_burst(std::span<const uint32_t> data) override;
        virtual VoidResult read(uint32_t& data) override;
        virtual VoidResult read_burst(std::span<uint32_t> data) override;

        const PIO  block;
        const uint node;

       protected:
        virtual VoidResult on_load() override;
        virtual VoidResult on_unload() override;

        VoidResult preloadReg(pio_src_dest reg, uint32_t value);

        Program::Ptr      program;
        Dma::Device::List dmalist;
        static Mutex      claim_mutex;
    };

}  // namespace corekit::Pio