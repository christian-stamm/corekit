#include "corekit/assert.hpp"
#include "corekit/executor.hpp"
#include "corekit/logger.hpp"
#include "corekit/stoptoken.hpp"
#include "corekit/task.hpp"
#include "corekit/time.hpp"

using namespace corekit;

class HelloTask : public Task {
   public:
    virtual VoidResult on_run(StopToken token) override {
        logger_.info() << "Hello from HelloTask!";
        return VoidResult();
    }

   private:
    Logger logger_{"HelloTask"};
};

class WorldTask : public Task {
   public:
    virtual VoidResult on_run(StopToken token) override {
        logger_.info() << "Hello from WorldTask!";
        return VoidResult();
    }

   private:
    Logger logger_{"WorldTask"};
};

class Spawner : public Task {
   public:
    Spawner(Executor& executor) : executor_(executor) {}

    virtual VoidResult on_run(StopToken token) override {
        for (int i = 0; i < 5; ++i) {
            HelloTask::Ptr hello_task = std::make_shared<HelloTask>();
            WorldTask::Ptr world_task = std::make_shared<WorldTask>();

            executor_.enqueue(hello_task);
            executor_.enqueue(world_task);

            logger_.info() << "Enqueued tasks.";
            Time::sleep(2);
        }

        return VoidResult();
    }

   private:
    Executor& executor_;
    Logger    logger_{"Spawner"};
};

int main() {
    Logger   logger{"Main"};
    Executor executor(1, 10);

    executor.enqueue(std::make_shared<Spawner>(executor));

    logger.info() << "Starting executor...";
    executor.launch();
    logger.info() << "Executor finished.";

    return 0;
}