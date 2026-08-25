#pragma once
#include <memory>

#include "corekit/atomic.hpp"
#include "corekit/types.hpp"
#include "corekit/watch.hpp"

namespace corekit {

    class BaseDevice {
       public:
        using Ptr = std::shared_ptr<BaseDevice>;

        BaseDevice(const Name& name);
        BaseDevice(const BaseDevice& other)            = delete;
        BaseDevice& operator=(const BaseDevice& other) = delete;

        virtual ~BaseDevice();

        bool   load();
        bool   unload();
        bool   reload();
        bool   isLoaded() const;
        double uptime() const;

        const Name name;

       protected:
        virtual bool onLoad() {
            return true;
        };

        virtual bool onUnload() {
            return true;
        };

       private:
        Atomic<bool> loaded;
        Watch        watch;
    };

};  // namespace corekit
