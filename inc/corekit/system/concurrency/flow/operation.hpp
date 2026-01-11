#pragma once
#include <concepts>
#include <memory>
#include <tuple>

#include "corekit/system/concurrency/flow/receiver.hpp"
#include "corekit/system/concurrency/queue.hpp"

namespace corekit {
    namespace system {
        namespace concurrency {

            struct Operation {
                friend class Executor;
                using Ptr = std::shared_ptr<Operation>;

                virtual ~Operation() = default;
                virtual void exec()  = 0;

                bool isDone() const {
                    return done;
                }

                bool isBusy() const {
                    return busy;
                }

                bool hasError() const {
                    return error;
                }

               protected:
                bool done  = false;
                bool busy  = false;
                bool error = false;
            };

            template <typename Fn, typename... Args>
                requires std::invocable<Fn, Args...>
            struct Task : public Operation {
                friend class Executor;

                using Ptr         = std::shared_ptr<Task<Fn, Args...>>;
                using Result      = std::invoke_result_t<Fn, Args...>;
                using Subscribers = std::vector<Receiver<Result>>;

                void subscribe(const Receiver<Result>& receiver) {
                    subs.push_back(receiver);
                }

                void exec() override {
                    try {
                        const Result& result = std::apply(work, args);

                        for (const auto& subscriber : subs) {
                            if (subscriber.notifier != nullptr) {
                                subscriber.notifier(result);
                            }
                        }

                    } catch (const std::exception& e) {
                        this->error = true;

                        for (const auto& subscriber : subs) {
                            if (subscriber.interrupt != nullptr) {
                                subscriber.interrupt(e);
                            }
                        }
                    }
                }

               protected:
                Subscribers         subs;
                Fn                  work;
                std::tuple<Args...> args;
            };

            using TaskList = SafeQueue<Operation::Ptr>;

        };  // namespace concurrency
    };  // namespace system
};  // namespace corekit
