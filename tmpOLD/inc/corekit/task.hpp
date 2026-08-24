#pragma once
#include <any>
#include <concepts>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>

#include "corekit/utils/queue.hpp"

namespace corekit {
    namespace system {
        class Executor;
    };
};  // namespace corekit

namespace corekit {
    namespace utils {

        using namespace corekit::system;

        // Type-erased callable wrapper
        class CallableWrapper {
           public:
            template <typename Fn>
            CallableWrapper(Fn&& fn)
                : callable(std::make_shared<CallableModel<std::decay_t<Fn>>>(
                      std::forward<Fn>(fn))) {}

            CallableWrapper() : callable(nullptr) {}

            template <typename... Args>
            auto invoke(Args&&... args) {
                if (callable) {
                    return callable->invoke_impl(std::forward<Args>(args)...);
                }
                throw std::runtime_error("CallableWrapper is empty");
            }

           private:
            struct CallableBase {
                virtual ~CallableBase()        = default;
                virtual std::any invoke_impl() = 0;
            };

            template <typename Fn>
            struct CallableModel : CallableBase {
                Fn fn;

                CallableModel(Fn&& f) : fn(std::forward<Fn>(f)) {}

                std::any invoke_impl() override {
                    return std::any(fn());
                }
            };

            std::shared_ptr<CallableBase> callable;
        };

        struct Operation {
            friend class corekit::system::Executor;
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

        template <typename T>
        struct Receiver {
            using KillEvent   = std::function<void(const std::exception&)>;
            using NotifyEvent = std::function<void(const T& result)>;

            NotifyEvent notifier  = nullptr;
            KillEvent   interrupt = nullptr;
        };

        template <>
        struct Receiver<void> {
            using KillEvent   = std::function<void(const std::exception&)>;
            using NotifyEvent = std::function<void()>;

            NotifyEvent notifier  = nullptr;
            KillEvent   interrupt = nullptr;
        };

        template <typename Fn, typename... Args>
        requires std::invocable<Fn, Args...>
        struct Task : public Operation {
            friend class corekit::system::Executor;

            using Ptr         = std::shared_ptr<Task<Fn, Args...>>;
            using Result      = std::invoke_result_t<Fn, Args...>;
            using Subscribers = std::vector<Receiver<Result>>;

            Task() = default;

            void subscribe(const Receiver<Result>& receiver) {
                subs.push_back(receiver);
            }

            void exec() override {
                try {
                    if constexpr (std::is_void_v<Result>) {
                        std::apply(work, args);

                        for (const auto& subscriber : subs) {
                            if (subscriber.notifier != nullptr) {
                                subscriber.notifier();
                            }
                        }
                    } else {
                        const Result& result = std::apply(work, args);

                        for (const auto& subscriber : subs) {
                            if (subscriber.notifier != nullptr) {
                                subscriber.notifier(result);
                            }
                        }
                    }

                    this->done = true;

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
            Subscribers                    subs;
            std::function<Result(Args...)> work;
            std::tuple<Args...>            args;
        };

        using TaskList = SafeQueue<Operation::Ptr>;

    };  // namespace utils
};      // namespace corekit
