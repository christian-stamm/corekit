#pragma once

namespace corekit {
    namespace system {

        namespace concurrency {

            class Task {
               public:
                Task();
                ~Task();

                void exec();
                bool done() const;

               private:
                struct Impl;
                Impl* pimpl;
            };

        };  // namespace concurrency

    };  // namespace system
};  // namespace corekit
