#pragma once

#include <hardware/pio.h>

#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <optional>

#include "corekit/async.hpp"

extern bool pio_sm_is_enabled(PIO block, uint sm);

namespace corekit {

    namespace Pio {
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

            virtual bool install(PIO block) final;
            virtual bool uninstall(PIO block) final;
            virtual bool isInstalled(PIO block) const final;
            virtual bool modify(PIO block, uint line, Command command) final;

            virtual bool registerNode(PIO block, uint node) final;
            virtual bool unregisterNode(PIO block, uint node) final;

            virtual NodeConf   buildNodeConf(PIO block, uint node) = 0;
            virtual LaunchConf buildLaunchConf(PIO block, uint node);

            virtual const State& getState(PIO block) const final;

           private:
            virtual State& requestState(PIO block) const final;

            mutable std::map<PIO, State> states;
        };

        template <typename T>
        class Node : public AsyncDevice<T> {
           public:
            using Ptr = std::shared_ptr<Node>;

            Node(const PIO block, uint node);
            virtual ~Node() override;

            Ptr requestUnused(const PIO block);

            bool deploy(const Program::Ptr& program);
            bool isRunning() const;

            virtual bool write(const T& data) override;
            virtual bool writeBulk(std::span<const T> data) override;
            virtual bool read(T& data) override;
            virtual bool readBulk(std::span<T> data) override;

            std::string getName() const {
                const std::string pioName =
                    std::to_string(pio_get_index(block));
                const std::string nodeName = std::to_string(node);

                return "PIO" + pioName + "-" + nodeName;
            }

            const PIO  block;
            const uint node;

           protected:
            virtual bool onLoad() override;
            virtual bool onUnload() override;

            void preloadReg(pio_src_dest reg, uint32_t value);

            Program::Ptr program;
        };
    }  // namespace Pio

}  // namespace corekit