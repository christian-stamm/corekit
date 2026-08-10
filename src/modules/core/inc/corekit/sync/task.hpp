#include <concepts>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "corekit/queue.hpp"

class ITask {
public:
    using Ptr = std::shared_ptr<ITask>;
    virtual ~ITask() = default;
    virtual void run() = 0;

    bool isDone() const { return done; }
    bool isBusy() const { return busy; }
    bool hasError() const { return error; }

    protected:
    bool done = false;
    bool busy = false;
    bool error = false;
};

template <typename Result>
struct CallbackTraits {
    using Type = std::function<void(const Result&)>;
};

template <>
struct CallbackTraits<void> {
    using Type = std::function<void()>;
};

template <typename Work>
    requires std::invocable<Work&>
class Task final : public ITask {
public:
    using Ptr = std::shared_ptr<Task<Work>>;
    using Result = std::invoke_result_t<Work&>;
    using Callback = typename CallbackTraits<Result>::Type;

    Task(Work &&work)
        : work(std::forward<Work>(work))
    {}

    void run() override
    {
        busy = true;
        done = false;
        
        try {
            if constexpr (std::is_void_v<Result>) {
                work();
                busy = false;
                notify();
            } else {
                auto result = work();
                busy = false;
                notify(result);
            }
        } catch (...) {
            error = true;
        }

        busy = false;
        done = true;
    }

    template <typename Callback>
        requires std::is_void_v<Result> && std::invocable<Callback&>
    Task& then(Callback&& callback)
    {
        notifier_.emplace_back(std::forward<Callback>(callback));
        return *this;
    }

    template <typename Callback>
        requires (!std::is_void_v<Result>) && std::invocable<Callback&, const Result&>
    Task& then(Callback&& callback)
    {
        notifier_.emplace_back(std::forward<Callback>(callback));
        return *this;
    }

    

private:
    void notify()
    {
        for (auto& callback : notifier_) {
            callback();
        }
    }

    template <typename Result>
    void notify(const Result& result)
    {
        for (auto& callback : notifier_) {
            callback(result);
        }
    }

    Work work;
    std::vector<Callback> notifier_;
};



class TaskQueue : private corekit::SafeQueue<ITask::Ptr> {
public:
    explicit TaskQueue(size_t capacity)
        : corekit::SafeQueue<ITask::Ptr>(capacity)
    {}

    template <typename F, typename... Args>
        requires std::invocable<F&, Args&...>
    auto enqueue(F&& function, Args&&... args)
    {
        auto work = std::bind_front(
           std::forward<F>(function),
            std::forward<Args>(args)...
        );

        using TaskType = Task<decltype(work)>;
        auto task = std::make_shared<TaskType>(
            std::move(work)
        );

        if(!this->try_push(task)) {
            task.reset();
        }

        return task;
    }

    void run()
    {
        ITask::Ptr task = nullptr;
        
        while (this->try_pop(task)) {
            if (task) {
                task->run();
            }
        }
    }
};