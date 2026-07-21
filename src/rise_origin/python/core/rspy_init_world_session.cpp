// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "core/rspy_init_world_session.h"

void initRSPYWorldSession(py::module &m) {
    /*****************************************************************
     * World session
     *****************************************************************/
    py::enum_<RSE_WorldSessionScope>(
        m, "RSE_WorldSessionScope", "Access scope a session grants to a component."
    )
        .value("RSE_WORLD_SESSION_SCOPE_OWNED", RSE_WORLD_SESSION_SCOPE_OWNED)
        .value("RSE_WORLD_SESSION_SCOPE_FULL", RSE_WORLD_SESSION_SCOPE_FULL)
        .export_values();

    py::class_<RS_WorldSession>(
        m, "RS_WorldSession", "A session over a running world's handle and callback."
    )
        .def(
            py::init([](RSPY_WorldHandle &handle, RSPY_WorldCallback &callback) {
                return std::make_unique<RS_WorldSession>(handle.handle, callback.callback);
            }),
            py::arg("handle"),
            py::arg("callback"),
            R"doc(
            Build a session over a running world's handle and callback.
        )doc"
        )
        .def(
            "has_component",
            &RS_WorldSession::hasComponent,
            py::arg("name"),
            R"doc(
            Whether a component is registered under the given name.
        )doc"
        )
        .def(
            "add_component",
            &RS_WorldSession::addComponent,
            py::arg("name"),
            py::arg("component"),
            py::arg("scope") = RSE_WORLD_SESSION_SCOPE_OWNED,
            R"doc(
            Attach a component under a name, giving it its own scoped access. The scope decides the
            component's power over the world: OWNED is the default sandbox, FULL grants read and
            write over the whole world.
        )doc"
        )
        .def(
            "remove_component",
            &RS_WorldSession::removeComponent,
            py::arg("name"),
            R"doc(
            Detach and remove the named component.
        )doc"
        )
        .def(
            "get_core_component",
            &RS_WorldSession::getCoreComponent,
            R"doc(
            The Core component, which owns the initial world and carries the user callback.
        )doc"
        )
        .def(
            "get_handle",
            [](RS_WorldSession &session) { return RSPY_WorldHandle(session.getHandle()); },
            R"doc(
            A fresh scoped handle to drive lifecycle (terminate, wait_for_end, ...).
        )doc"
        )
        .def(
            "run",
            &RS_WorldSession::run,
            py::arg("is_blocking") = true,
            py::call_guard<py::gil_scoped_release>(),
            R"doc(
            When blocking, wait until the world ends and then detach all components; otherwise
            return at once and leave teardown to the destructor.
        )doc"
        );
}
