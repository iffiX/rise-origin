// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_SESSION_H
#define RS_WORLD_SESSION_H

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rise_origin/core/rs_world_callback.h"
#include "rise_origin/core/rs_world_controller.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/rsc/rsc_world.h"
#include "rise_origin/utils/concurrency/rs_thread_pool.h"
#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"

namespace rise {

/**
 * @brief Thrown when a component touches an object outside its capability scope.
 * @details A component may read anything but may only modify or delete what it created.
 */
RS_MAKE_EXCEPTION(RS_WorldSessionScopeException, RS_InvalidAccessException);

/**
 * @brief Channel name for a broker message; the empty string is the broadcast channel.
 */
inline const std::string RS_WORLD_SESSION_BROADCAST{};

/**
 * @brief In-process, zero-copy, type-and-channel keyed publish/subscribe bus for cross-component
 * coordination.
 * @details A message is a plain struct delivered as shared_ptr<const T>; publish enqueues a
 * small envelope into every matching subscriber's queue, and the subscriber drains it on its own
 * thread. The broker is template based, so it lives entirely in this header.
 */
class RS_WorldSessionMessageBroker {
   private:
    // Internal types first: the public Subscription names them in its data members, and a member
    // declaration is not a complete-class context, so they must precede it lexically.
    struct Key {
        std::type_index type;
        std::string channel;
        bool operator==(const Key &other) const {
            return type == other.type && channel == other.channel;
        }
    };
    struct KeyHash {
        RSize operator()(const Key &key) const {
            return std::hash<std::type_index>{}(key.type) ^
                   (std::hash<std::string>{}(key.channel) << 1);
        }
    };
    // The type-erased, zero-copy envelope that travels the bus: a type tag plus the message.
    struct MessageEnvelope {
        std::type_index type = std::type_index(typeid(void));
        std::shared_ptr<const void> message;
    };
    // Per-subscriber multi-producer queue of envelopes.
    class SessionQueue {
       public:
        void push(MessageEnvelope envelope) {
            std::scoped_lock lock(_mutex);
            _queue.push_back(std::move(envelope));
        }
        bool pop(MessageEnvelope &out) {
            std::scoped_lock lock(_mutex);
            if (_queue.empty()) return false;
            out = std::move(_queue.front());
            _queue.pop_front();
            return true;
        }

       private:
        std::mutex _mutex;
        std::deque<MessageEnvelope> _queue;
    };

   public:
    /**
     * @brief RAII subscription handle. The owner drains it on its own thread to receive messages
     * and unsubscribes on destruction.
     */
    class Subscription {
       public:
        Subscription() = default;
        Subscription(Subscription &&other) noexcept { _moveFrom(other); }
        Subscription &operator=(Subscription &&other) noexcept {
            if (this != &other) {
                reset();
                _moveFrom(other);
            }
            return *this;
        }
        Subscription(const Subscription &) = delete;
        Subscription &operator=(const Subscription &) = delete;
        ~Subscription() { reset(); }

        /**
         * @brief Dispatch every queued message to the callback. Call from the subscriber's thread.
         */
        void drain() {
            if (!_queue) return;
            MessageEnvelope envelope;
            while (_queue->pop(envelope)) _dispatch(envelope);
        }

        /**
         * @brief Unsubscribe and release the queue.
         */
        void reset() {
            if (_broker) _broker->_unsubscribe(_key, _id);
            _broker = nullptr;
            _queue = nullptr;
            _dispatch = nullptr;
        }

       private:
        friend class RS_WorldSessionMessageBroker;
        Subscription(
            RS_WorldSessionMessageBroker *broker,
            Key key,
            RIndex id,
            std::shared_ptr<SessionQueue> queue,
            std::function<void(const MessageEnvelope &)> dispatch
        )
            : _broker(broker),
              _key(std::move(key)),
              _id(id),
              _queue(std::move(queue)),
              _dispatch(std::move(dispatch)) {}

        void _moveFrom(Subscription &other) {
            _broker = other._broker;
            _key = other._key;
            _id = other._id;
            _queue = std::move(other._queue);
            _dispatch = std::move(other._dispatch);
            other._broker = nullptr;
        }

        RS_WorldSessionMessageBroker *_broker = nullptr;
        Key _key{std::type_index(typeid(void)), {}};
        RIndex _id = RS_NULL_INDEX;
        std::shared_ptr<SessionQueue> _queue;
        std::function<void(const MessageEnvelope &)> _dispatch;
    };

    /**
     * @brief Publish a message to every subscriber registered on (T, channel).
     */
    template <typename T>
    void publish(const std::string &channel, std::shared_ptr<const T> message) {
        Key key{std::type_index(typeid(T)), channel};
        std::shared_lock lock(_registry_mutex);
        auto found = _queues.find(key);
        if (found == _queues.end()) return;
        for (auto &[id, queue] : found->second) {
            queue->push(
                {std::type_index(typeid(T)), std::static_pointer_cast<const void>(message)}
            );
        }
    }

    /**
     * @brief Subscribe a typed callback to (T, channel); returns an RAII subscription.
     */
    template <typename T>
    Subscription
    subscribe(const std::string &channel, std::function<void(std::shared_ptr<const T>)> callback) {
        Key key{std::type_index(typeid(T)), channel};
        auto queue = std::make_shared<SessionQueue>();
        RIndex id;
        {
            std::unique_lock lock(_registry_mutex);
            id = _next_id++;
            _queues[key][id] = queue;
        }
        auto dispatch = [callback = std::move(callback)](const MessageEnvelope &envelope) {
            callback(std::static_pointer_cast<const T>(envelope.message));
        };
        return Subscription(this, std::move(key), id, std::move(queue), std::move(dispatch));
    }

   private:
    std::shared_mutex _registry_mutex;
    std::unordered_map<Key, std::unordered_map<RIndex, std::shared_ptr<SessionQueue>>, KeyHash>
        _queues;
    RIndex _next_id = 0;

    void _unsubscribe(const Key &key, RIndex id) {
        std::unique_lock lock(_registry_mutex);
        if (auto found = _queues.find(key); found != _queues.end()) found->second.erase(id);
    }
};

/**
 * @brief A scoped world handle that limits which objects a component may change.
 * @details It owns the set of object names for this scope, seeded at construction. A component may
 * read any object in the world. It may only modify or delete the objects it owns. Changing anything
 * else throws RS_WorldSessionScopeException.
 */
class RS_WorldSessionHandle : public RS_WorldHandle {
   public:
    explicit RS_WorldSessionHandle(
        std::shared_ptr<RS_WorldHandle> raw,
        std::unordered_set<std::string> owned_resources = {},
        std::unordered_set<std::string> owned_essences = {},
        std::unordered_set<std::string> owned_entities = {},
        std::unordered_set<std::string> owned_records = {},
        std::unordered_set<std::string> owned_signals = {},
        std::unordered_set<std::string> owned_recorders = {},
        std::unordered_set<std::string> owned_controllers = {}
    )
        : _raw(std::move(raw)),
          _owned_resources(std::move(owned_resources)),
          _owned_essences(std::move(owned_essences)),
          _owned_entities(std::move(owned_entities)),
          _owned_records(std::move(owned_records)),
          _owned_signals(std::move(owned_signals)),
          _owned_recorders(std::move(owned_recorders)),
          _owned_controllers(std::move(owned_controllers)) {}

    // Identity / state / messages / lifecycle (global reads, forwarded).
    std::string name() override;
    std::string getInfo(const std::string &info_name) override;
    bool isPending() override;
    bool isRunning() override;
    bool isPaused() override;
    bool isEnd() override;
    void pause() override;
    void resume() override;
    void terminate() override;
    RSE_WorldStatus getStatus() override;
    bool waitForRunning() override;
    void waitForEnd() override;
    RBigIndex getNewestMessageIndex() override;
    RBigIndex getOldestMessageIndex() override;
    std::vector<RS_WorldMessage> getMessagesMatching(
        RSize max_count = RS_MAX_SIZE,
        RBigIndex oldest_index = 0,
        RBigIndex newest_index = RS_NULL_BIG_INDEX,
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO
    ) override;
    RS_WorldMessage waitForMessage(RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO) override;
    RSize getMessageBufferSize() override;
    void setMessageBufferSize(RSize max_messages) override;

    // Modification transaction: forwarded (the raw handle holds its lock across the span).
    void beginModificationTransaction() override;
    void endModificationTransaction() override;

    // Root (read allowed; setSimulation is global).
    rsc::RSC_World getWorld() override;
    RS_WorldTask setSimulation(const rsc::RSC_Simulation &simulation) override;

    // Resource.
    RS_WorldTask addResource(const rsc::RSC_ResourceVariant &resource) override;
    RS_WorldTask removeResource(const std::string &name) override;
    rsc::RSC_ResourceVariant findResource(const std::string &name) override;

    // Essence.
    RS_WorldTask addEssence(const rsc::RSC_EssenceVariant &essence) override;
    RS_WorldTask
    updateEssence(const std::string &name, const rsc::RSC_EssenceVariant &essence) override;
    RS_WorldTask removeEssence(const std::string &name) override;
    rsc::RSC_EssenceVariant findEssence(const std::string &name) override;

    // Entity.
    RS_WorldTask addEntity(const rsc::RSC_EntityVariant &entity) override;
    RS_WorldTask removeEntity(const std::string &name) override;
    RS_WorldTask freezeEntity(const std::string &name) override;
    RS_WorldTask unfreezeEntity(const std::string &name) override;
    rsc::RSC_EntityVariant findEntity(const std::string &name) override;
    bool isEntityFrozen(const std::string &name) override;
    RIndex findEntityGID(const std::string &name) override;
    std::string findEntityName(RIndex gid) override;

    // Record.
    RS_WorldTask addRecord(const rsc::RSC_RecordVariant &record) override;
    RS_WorldTask removeRecord(const std::string &name) override;
    rsc::RSC_RecordVariant findRecord(const std::string &name) override;

    // Signal.
    RS_WorldTask addSignal(const rsc::RSC_SignalVariant &signal) override;
    RS_WorldTask removeSignal(const std::string &name) override;
    rsc::RSC_SignalVariant findSignal(const std::string &name) override;

    // Recorder.
    RS_WorldTask addRecorder(const rsc::RSC_RecorderVariant &recorder) override;
    RS_WorldTask removeRecorder(const std::string &name) override;
    rsc::RSC_RecorderVariant findRecorder(const std::string &name) override;

    // Controller.
    RS_WorldTask addController(const rsc::RSC_ControllerVariant &controller) override;
    RS_WorldTask removeController(const std::string &name) override;
    rsc::RSC_ControllerVariant findController(const std::string &name) override;

   private:
    // The access reads this scope's owned controllers.
    friend class RS_WorldSessionAccess;

    template <typename, typename = void>
    struct HasName : std::false_type {};
    template <typename T>
    struct HasName<T, std::void_t<decltype(std::declval<const T &>().name)>> : std::true_type {};

    std::shared_ptr<RS_WorldHandle> _raw;
    std::unordered_set<std::string> _owned_resources;
    std::unordered_set<std::string> _owned_essences;
    std::unordered_set<std::string> _owned_entities;
    std::unordered_set<std::string> _owned_records;
    std::unordered_set<std::string> _owned_signals;
    std::unordered_set<std::string> _owned_recorders;
    std::unordered_set<std::string> _owned_controllers;

    static void
    _requireOwned(const std::string &name, const std::unordered_set<std::string> &owned);

    template <typename Variant>
    static std::string _nameOf(const Variant &variant) {
        return variant.visit([](const auto &held) -> std::string {
            using Held = std::decay_t<decltype(held)>;
            if constexpr (HasName<Held>::value) {
                return held.name;
            } else {
                return {};
            }
        });
    }
};

/**
 * @brief What a component uses each step: its scoped handle, this step's controllers, and the
 * broker.
 */
class RS_WorldSessionAccess {
   public:
    RS_WorldSessionAccess(
        std::shared_ptr<RS_WorldSessionHandle> handle,
        std::vector<RS_WorldController> &controllers,
        RS_WorldSessionMessageBroker &broker
    )
        : _handle(std::move(handle)), _controllers(controllers), _broker(broker) {}

    /**
     * @brief The scoped handle for this component.
     */
    [[nodiscard]] std::shared_ptr<RS_WorldSessionHandle> handle() const { return _handle; }

    /**
     * @brief This step's controllers that this scope owns.
     */
    [[nodiscard]] std::vector<std::reference_wrapper<RS_WorldController>> controllers();

    /**
     * @brief The named controller in this scope for this step.
     */
    RS_WorldController &controller(const std::string &name);

    /**
     * @brief Publish a message on the cross-component broker.
     */
    template <typename T>
    void publish(const std::string &channel, std::shared_ptr<const T> message) {
        _broker.get().publish(channel, std::move(message));
    }

    /**
     * @brief Subscribe to cross-component messages on the broker.
     */
    template <typename T>
    RS_WorldSessionMessageBroker::Subscription
    subscribe(const std::string &channel, std::function<void(std::shared_ptr<const T>)> callback) {
        return _broker.get().subscribe<T>(channel, std::move(callback));
    }

   private:
    std::shared_ptr<RS_WorldSessionHandle> _handle;
    std::reference_wrapper<std::vector<RS_WorldController>> _controllers;
    std::reference_wrapper<RS_WorldSessionMessageBroker> _broker;
};

/**
 * @brief A session participant. Components manifest presence in onAttach, push this step's work
 * onto the shared pool in onCallback, and tear down in onDetach.
 */
class RS_WorldSessionComponent {
   public:
    virtual ~RS_WorldSessionComponent() = default;

    /**
     * @brief Take ownership of the component's access; manifest presence; start threads.
     */
    virtual void onAttach(RS_WorldSessionAccess access) = 0;

    /**
     * @brief React to a control step by pushing this step's jobs onto the shared pool and returning
     * without blocking. The session waits on the pool once after every component has pushed. Read
     * this scope's controllers through the stored access.
     */
    virtual void onCallback(RBigFloat time, RS_ThreadPool &pool) = 0;

    /**
     * @brief Tear down.
     */
    virtual void onDetach() = 0;
};

class RS_WorldSessionComponentGeneric;

/**
 * @brief Owns a running world's (handle, callback) and drives its components.
 * @details Keeps the raw handle private and hands each component its own scoped handle and access.
 * On construction it builds a Core component that owns the whole initial world. Binds its per-step
 * handler onto the callback and fans the step out to the Core plus the named components in
 * parallel.
 */
class RS_WorldSession {
   public:
    RS_WorldSession(
        std::shared_ptr<RS_WorldHandle> handle,
        std::shared_ptr<RS_WorldCallback> callback
    );
    ~RS_WorldSession();

    RS_WorldSession(const RS_WorldSession &) = delete;
    RS_WorldSession &operator=(const RS_WorldSession &) = delete;

    /**
     * @brief Whether a component is registered under the given name.
     */
    bool hasComponent(const std::string &name) const;

    /**
     * @brief Attach a component under a name, giving it its own scoped access. Called sequentially.
     */
    void addComponent(const std::string &name, std::shared_ptr<RS_WorldSessionComponent> component);

    /**
     * @brief Detach and remove the named component; throws RS_NotFoundException if absent.
     */
    void removeComponent(const std::string &name);

    /**
     * @brief The Core component, which owns the initial world and carries the user callback.
     */
    std::shared_ptr<RS_WorldSessionComponentGeneric> getCoreComponent();

    /**
     * @brief A fresh scoped access for non-component callers.
     */
    RS_WorldSessionAccess getAccess();

    /**
     * @brief A fresh scoped handle for non-component callers.
     */
    std::shared_ptr<RS_WorldSessionHandle> getHandle();

    /**
     * @brief When blocking, wait until the world ends and then detach all components; otherwise
     * return at once and leave teardown to the destructor.
     */
    void run(bool is_blocking = true);

   private:
    RS_ThreadPool _pool;
    std::shared_ptr<RS_WorldHandle> _raw;
    std::shared_ptr<RS_WorldCallback> _callback;
    RS_WorldSessionMessageBroker _broker;
    std::vector<RS_WorldController> _controllers;
    std::shared_ptr<RS_WorldSessionComponentGeneric> _core_component;
    std::unordered_map<std::string, std::shared_ptr<RS_WorldSessionComponent>> _components;
    bool _is_detached = false;

    RS_WorldSessionAccess _makeAccess();
    std::shared_ptr<RS_WorldSessionComponentGeneric> _makeCoreComponent();
    void _detachComponents();
    void _onCallback(RBigFloat time, std::vector<RS_WorldController> controllers);
};

}  // namespace rise

#endif  // RS_WORLD_SESSION_H
