#pragma once
#include <memory>

#include "corekit/atomic.hpp"
#include "corekit/result.hpp"
#include "corekit/watch.hpp"

namespace corekit {

    class BaseDevice {
       public:
        using Ptr = std::shared_ptr<BaseDevice>;

        BaseDevice(const std::string& name);
        BaseDevice(const BaseDevice& other)            = delete;
        BaseDevice& operator=(const BaseDevice& other) = delete;

        virtual ~BaseDevice();

        VoidResult load();
        VoidResult unload();
        VoidResult reload();
        bool       is_loaded() const;
        double     uptime() const;

        const std::string name;

       protected:
        virtual VoidResult on_load() {
            return VoidResult();
        };

        virtual VoidResult on_unload() {
            return VoidResult();
        };

       private:
        Atomic<bool> loaded;
        Watch        watch;
    };

};  // namespace corekit
