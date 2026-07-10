// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "core/init_world_handle.h"

void initRSPYWorldHandle(py::module &m) {
    /*****************************************************************
     * Enums
     *****************************************************************/
    py::enum_<RSE_WorldMessageLevel>(m, "RSE_WorldMessageLevel")
        .value("RSE_WORLD_MESSAGE_INFO", RSE_WORLD_MESSAGE_INFO)
        .value("RSE_WORLD_MESSAGE_WARNING", RSE_WORLD_MESSAGE_WARNING)
        .value("RSE_WORLD_MESSAGE_ERROR", RSE_WORLD_MESSAGE_ERROR)
        .export_values();

    py::enum_<RSE_WorldTaskState>(m, "RSE_WorldTaskState")
        .value("RSE_WORLD_TASK_PENDING", RSE_WORLD_TASK_PENDING)
        .value("RSE_WORLD_TASK_SUCCESS", RSE_WORLD_TASK_SUCCESS)
        .value("RSE_WORLD_TASK_FAIL", RSE_WORLD_TASK_FAIL)
        .export_values();

    /*****************************************************************
     * World messages
     *****************************************************************/
    py::class_<RS_WorldMessage>(m, "RS_WorldMessage", "One entry in a world's message log.")
        .def(py::init<>())
        .def_readonly(
            "index", &RS_WorldMessage::index, "Monotonic index of this message in the log."
        )
        .def_readonly("time", &RS_WorldMessage::time, "Timestamp when this message was recorded.")
        .def_readonly("level", &RS_WorldMessage::level, "Severity level of this message.")
        .def_readonly("text", &RS_WorldMessage::text, "Text content of this message.");

    /*****************************************************************
     * World tasks
     *****************************************************************/
    py::class_<RS_WorldTaskDataCreated>(
        m, "RS_WorldTaskDataCreated", "Task result carrying the gid a world add operation assigned."
    )
        .def(py::init<>())
        .def_readwrite(
            "gid", &RS_WorldTaskDataCreated::gid, "Global id the add operation assigned."
        )
        .def_readwrite("name", &RS_WorldTaskDataCreated::name, "Name of the created object.");

    py::class_<RS_WorldTaskDataSuccess>(
        m,
        "RS_WorldTaskDataSuccess",
        "Task result for a command that succeeded without producing an object."
    )
        .def(py::init<>());

    py::class_<RS_WorldTaskDataFailed>(
        m, "RS_WorldTaskDataFailed", "Task result for a command that failed."
    )
        .def(py::init<>())
        .def_readwrite(
            "error", &RS_WorldTaskDataFailed::error, "Error message describing the failure."
        );

    py::class_<RS_WorldTaskData>(m, "RS_WorldTaskData", "Result body carried by a world task.")
        .def(py::init<>())
        .def_property_readonly(
            "value",
            [](const RS_WorldTaskData &self) -> py::object {
                if (self.index() == RS_NULL_INDEX) return py::none();
                py::object result = py::none();
                self.visit([&](auto &held) { result = py::cast(held); });
                return result;
            },
            "The held task result, or None when empty."
        );

    py::class_<RS_WorldTask>(m, "RS_WorldTask", "Completion of a single deferred world command.")
        .def(py::init<>())
        .def("state", &RS_WorldTask::state, "Completion state without blocking.")
        .def(
            "is_finished",
            &RS_WorldTask::isFinished,
            "Whether the command has finished without blocking."
        )
        .def("is_valid", &RS_WorldTask::isValid, "Whether this task refers to a command.")
        .def(
            "wait",
            [](const RS_WorldTask &self) {
                py::gil_scoped_release release;
                self.wait();
            },
            "Block until the command finishes."
        )
        .def(
            "wait_for",
            [](const RS_WorldTask &self, double timeout_seconds) {
                py::gil_scoped_release release;
                return self.waitFor(std::chrono::duration<double>(timeout_seconds));
            },
            py::arg("timeout_seconds"),
            "Block until the command finishes or the timeout elapses."
        )
        .def(
            "get",
            [](const RS_WorldTask &self) {
                py::gil_scoped_release release;
                return self.get();
            },
            "Block until the command finishes and return its result."
        );

    /*****************************************************************
     * World handle
     *****************************************************************/

    py::class_<RSPY_WorldHandle>(m, "RS_WorldHandle")
        .def(
            "is_pending",
            &RSPY_WorldHandle::is_pending,
            R"doc(
            Reports whether the world is still uninitialized or initializing.

            Returns:
                True when the world has not yet started running.
        )doc"
        )
        .def(
            "is_running",
            &RSPY_WorldHandle::is_running,
            R"doc(
            Reports whether the world is currently running.

            Returns:
                True when the world is in the running state.
        )doc"
        )
        .def(
            "is_paused",
            &RSPY_WorldHandle::is_paused,
            R"doc(
            Reports whether the world is currently paused.

            Returns:
                True when the world is in the paused state.
        )doc"
        )
        .def(
            "is_end",
            &RSPY_WorldHandle::is_end,
            R"doc(
            Reports whether the world has ended.

            Returns:
                True when the world has reached any end state, whether clean or failed.
        )doc"
        )
        .def(
            "pause",
            &RSPY_WorldHandle::pause,
            R"doc(
            Pauses the running world.

            Blocks until the world is running, then requests a pause. Does nothing if the
            world ends before it starts running.
        )doc"
        )
        .def(
            "resume",
            &RSPY_WorldHandle::resume,
            R"doc(
            Resumes the paused world.

            Blocks until the world is running, then requests a resume. Does nothing if the
            world ends before it starts running.
        )doc"
        )
        .def(
            "terminate",
            &RSPY_WorldHandle::terminate,
            R"doc(
            Terminates the running world.

            Blocks until the world is running, then requests termination. Does nothing if
            the world ends before it starts running.
        )doc"
        )
        .def(
            "get_status",
            &RSPY_WorldHandle::get_status,
            R"doc(
            Reads the current world status.

            Returns:
                The current RSE_WorldStatus.
        )doc"
        )
        .def(
            "wait_for_running",
            &RSPY_WorldHandle::wait_for_running,
            R"doc(
            Blocks until the world starts running or ends.

            Releases the GIL while waiting.

            Returns:
                True if the world reached the running state, False if it ended first.
        )doc"
        )
        .def(
            "wait_for_end",
            &RSPY_WorldHandle::wait_for_end,
            R"doc(
            Blocks until the world ends.

            Releases the GIL while waiting.
        )doc"
        )
        .def(
            "get_newest_message_index",
            &RSPY_WorldHandle::get_newest_message_index,
            R"doc(
            Index of the newest message in the log, or RS_NULL_BIG_INDEX when empty.
        )doc"
        )
        .def(
            "get_oldest_message_index",
            &RSPY_WorldHandle::get_oldest_message_index,
            R"doc(
            Index of the oldest retained message; a cursor below it has missed entries.
        )doc"
        )
        .def(
            "get_messages_matching",
            &RSPY_WorldHandle::get_messages_matching,
            py::arg("max_count") = RS_MAX_SIZE,
            py::arg("oldest_index") = 0,
            py::arg("newest_index") = RS_NULL_BIG_INDEX,
            py::arg("level") = RSE_WORLD_MESSAGE_INFO,
            R"doc(
            Returns up to max_count messages in [oldest_index, newest_index) at severity >= level.

            Reads never remove. Results run oldest to newest; when capped, the newest are kept.
        )doc"
        )
        .def(
            "wait_for_message",
            &RSPY_WorldHandle::wait_for_message,
            py::arg("level") = RSE_WORLD_MESSAGE_INFO,
            R"doc(
            Blocks until the next message at severity >= level, then returns it.

            Releases the GIL while waiting.
        )doc"
        )
        .def(
            "get_message_buffer_size",
            &RSPY_WorldHandle::get_message_buffer_size,
            R"doc(
            Number of entries the message log retains.
        )doc"
        )
        .def(
            "set_message_buffer_size",
            &RSPY_WorldHandle::set_message_buffer_size,
            py::arg("max_messages"),
            R"doc(
            Sets the number of entries the message log retains; shrinking evicts the oldest.
        )doc"
        )
        .def(
            "begin_modification_transaction",
            &RSPY_WorldHandle::begin_modification_transaction,
            R"doc(
            Opens a modification transaction so several edits apply on the same step.

            Blocks until the world is running. Does nothing if the world ends before it
            starts running.
        )doc"
        )
        .def(
            "end_modification_transaction",
            &RSPY_WorldHandle::end_modification_transaction,
            R"doc(
            Closes the open modification transaction and applies its batched edits.

            Blocks until the world is running. Does nothing if the world ends before it
            starts running.
        )doc"
        )
        .def(
            "get_world",
            &RSPY_WorldHandle::get_world,
            R"doc(
            Snapshots the current world characterization.

            Blocks until the world is running, then reads back the live world including its
            active recorders and controllers.

            Returns:
                The current RSC_World, or an empty RSC_World if the world ended first.
        )doc"
        )
        .def(
            "set_simulation",
            &RSPY_WorldHandle::set_simulation,
            py::arg("simulation"),
            R"doc(
            Replaces the simulation parameters of the running world.

            Args:
                simulation: The RSC_Simulation to apply.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "add_resource",
            &RSPY_WorldHandle::add_resource,
            py::arg("resource"),
            R"doc(
            Validates and adds a resource to the running world.

            Args:
                resource: The RSC_ResourceVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the resource fails validation.
        )doc"
        )
        .def(
            "remove_resource",
            &RSPY_WorldHandle::remove_resource,
            py::arg("name"),
            R"doc(
            Removes the named resource from the running world.

            Args:
                name: The name of the resource to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_resource",
            &RSPY_WorldHandle::find_resource,
            py::arg("name"),
            R"doc(
            Looks up the named resource in the running world.

            Args:
                name: The name of the resource to find.

            Returns:
                The matching RSC_ResourceVariant, or an empty variant if the world is not
                running.
        )doc"
        )
        .def(
            "add_essence",
            &RSPY_WorldHandle::add_essence,
            py::arg("essence"),
            R"doc(
            Validates and adds an essence to the running world.

            Args:
                essence: The RSC_EssenceVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the essence fails validation.
        )doc"
        )
        .def(
            "update_essence",
            &RSPY_WorldHandle::update_essence,
            py::arg("name"),
            py::arg("essence"),
            R"doc(
            Validates and updates the named essence in the running world.

            Args:
                name: The name of the essence to update.
                essence: The RSC_EssenceVariant holding the new values.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the essence fails validation.
        )doc"
        )
        .def(
            "remove_essence",
            &RSPY_WorldHandle::remove_essence,
            py::arg("name"),
            R"doc(
            Removes the named essence from the running world.

            Args:
                name: The name of the essence to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_essence",
            &RSPY_WorldHandle::find_essence,
            py::arg("name"),
            R"doc(
            Looks up the named essence in the running world.

            Args:
                name: The name of the essence to find.

            Returns:
                The matching RSC_EssenceVariant, or an empty variant if the world is not
                running.
        )doc"
        )
        .def(
            "add_entity",
            &RSPY_WorldHandle::add_entity,
            py::arg("entity"),
            R"doc(
            Validates and adds an entity to the running world.

            Args:
                entity: The RSC_EntityVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the entity fails validation.
        )doc"
        )
        .def(
            "remove_entity",
            &RSPY_WorldHandle::remove_entity,
            py::arg("name"),
            R"doc(
            Removes the named entity from the running world.

            Args:
                name: The name of the entity to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "freeze_entity",
            &RSPY_WorldHandle::freeze_entity,
            py::arg("name"),
            R"doc(
            Freezes the named entity so it stops advancing.

            Args:
                name: The name of the entity to freeze.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "unfreeze_entity",
            &RSPY_WorldHandle::unfreeze_entity,
            py::arg("name"),
            R"doc(
            Unfreezes the named entity so it resumes advancing.

            Args:
                name: The name of the entity to unfreeze.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_entity",
            &RSPY_WorldHandle::find_entity,
            py::arg("name"),
            R"doc(
            Looks up the named entity in the running world.

            Args:
                name: The name of the entity to find.

            Returns:
                The matching RSC_EntityVariant, or an empty variant if the world is not
                running.
        )doc"
        )
        .def(
            "is_entity_frozen",
            &RSPY_WorldHandle::is_entity_frozen,
            py::arg("name"),
            R"doc(
            Reports whether the named entity is frozen.

            Args:
                name: The name of the entity to query.

            Returns:
                True if the entity is frozen, False if it is not or the world is not running.
        )doc"
        )
        .def(
            "add_record",
            &RSPY_WorldHandle::add_record,
            py::arg("record"),
            R"doc(
            Validates and adds a record to the running world.

            Args:
                record: The RSC_RecordVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the record fails validation.
        )doc"
        )
        .def(
            "remove_record",
            &RSPY_WorldHandle::remove_record,
            py::arg("name"),
            R"doc(
            Removes the named record from the running world.

            Args:
                name: The name of the record to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_record",
            &RSPY_WorldHandle::find_record,
            py::arg("name"),
            R"doc(
            Looks up the named record in the running world.

            Args:
                name: The name of the record to find.

            Returns:
                The matching RSC_RecordVariant, or an empty variant if the world is not running.
        )doc"
        )
        .def(
            "add_signal",
            &RSPY_WorldHandle::add_signal,
            py::arg("signal"),
            R"doc(
            Validates and adds a signal to the running world.

            Args:
                signal: The RSC_SignalVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the signal fails validation.
        )doc"
        )
        .def(
            "remove_signal",
            &RSPY_WorldHandle::remove_signal,
            py::arg("name"),
            R"doc(
            Removes the named signal from the running world.

            Args:
                name: The name of the signal to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_signal",
            &RSPY_WorldHandle::find_signal,
            py::arg("name"),
            R"doc(
            Looks up the named signal in the running world.

            Args:
                name: The name of the signal to find.

            Returns:
                The matching RSC_SignalVariant, or an empty variant if the world is not running.
        )doc"
        )
        .def(
            "add_recorder",
            &RSPY_WorldHandle::add_recorder,
            py::arg("recorder"),
            R"doc(
            Validates and adds a recorder to the running world.

            Args:
                recorder: The RSC_RecorderVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the recorder fails validation.
        )doc"
        )
        .def(
            "remove_recorder",
            &RSPY_WorldHandle::remove_recorder,
            py::arg("name"),
            R"doc(
            Removes the named recorder from the running world.

            Args:
                name: The name of the recorder to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_recorder",
            &RSPY_WorldHandle::find_recorder,
            py::arg("name"),
            R"doc(
            Looks up the named recorder in the running world.

            Args:
                name: The name of the recorder to find.

            Returns:
                The matching RSC_RecorderVariant, or an empty variant if the world is not
                running.
        )doc"
        )
        .def(
            "add_controller",
            &RSPY_WorldHandle::add_controller,
            py::arg("controller"),
            R"doc(
            Validates and adds a controller to the running world.

            Args:
                controller: The RSC_ControllerVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the controller fails validation.
        )doc"
        )
        .def(
            "remove_controller",
            &RSPY_WorldHandle::remove_controller,
            py::arg("name"),
            R"doc(
            Removes the named controller from the running world.

            Args:
                name: The name of the controller to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_controller",
            &RSPY_WorldHandle::find_controller,
            py::arg("name"),
            R"doc(
            Looks up the named controller in the running world.

            Args:
                name: The name of the controller to find.

            Returns:
                The matching RSC_ControllerVariant, or an empty variant if the world is not
                running.
        )doc"
        );
}
