#pragma once
#include <memory>

#include "corekit/atomic.hpp"
#include "corekit/watch.hpp"

namespace corekit {

    class BaseDevice {
       public:
        using Ptr = std::shared_ptr<BaseDevice>;

        BaseDevice(const std::string& name);
        BaseDevice(const BaseDevice& other)            = delete;
        BaseDevice& operator=(const BaseDevice& other) = delete;

        virtual ~BaseDevice();

        bool   load();
        bool   unload();
        bool   reload();
        bool   isLoaded() const;
        double uptime() const;

        const std::string name;

       protected:
        virtual bool on_load() {
            return true;
        };

        virtual bool on_unload() {
            return true;
        };

       private:
        Atomic<bool> loaded;
        Watch        watch;
    };

};  // namespace corekit
