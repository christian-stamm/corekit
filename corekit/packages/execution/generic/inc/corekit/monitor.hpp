#pragma once

#include <memory>

#include "corekit/task.hpp"

namespace corekit {

    class Monitor : public Task {
       public:
        using Ptr = std::shared_ptr<Monitor>;

        Monitor();

       protected:
        VoidResult on_run(StopToken token) override;
    };

}  // namespace corekit