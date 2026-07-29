// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "core/rspy_init_world_session_component.h"

void initRSPYWorldSessionComponent(py::module &m) {
    /*****************************************************************
     * World session components
     *****************************************************************/

    py::class_<RS_WorldSessionComponent, std::shared_ptr<RS_WorldSessionComponent>>(
        m,
        "RS_WorldSessionComponent",
        "A session participant that manifests presence, pushes work each step, and tears down."
    );

    py::class_<
        RS_WorldSessionComponentGeneric,
        RS_WorldSessionComponent,
        std::shared_ptr<RS_WorldSessionComponentGeneric>>(m, "RS_WorldSessionComponentGeneric")
        .def(
            py::init<>(),
            R"doc(
            A component that runs a user callback over the controllers in its scope each step.
        )doc"
        )
        .def(
            "get_handle",
            [](RS_WorldSessionComponentGeneric &component) {
                return RSPY_WorldHandle(component.getHandle());
            },
            R"doc(
            This component's scoped world handle.
        )doc"
        )
        .def(
            "set_callback",
            [](RS_WorldSessionComponentGeneric &component, py::function py_callback) {
                // The callback fires from a pool thread, so its Python object is held under a
                // deleter that takes the GIL, and each call acquires the GIL before invoking it.
                std::shared_ptr<py::object> py_object(
                    new py::object(std::move(py_callback)), [](py::object *object) {
                        py::gil_scoped_acquire gil;
                        delete object;
                    }
                );
                component.setCallback(
                    [py_object](
                        RBigFloat time,
                        std::vector<std::reference_wrapper<RS_WorldController>> controllers
                    ) {
                        py::gil_scoped_acquire gil;
                        py::list controllers_list;
                        for (auto &controller : controllers) {
                            controllers_list.append(RSPY_WorldController(controller.get()));
                        }
                        try {
                            py::cast<py::function> (*py_object)(time, controllers_list);
                        } catch (const py::error_already_set &error) {
                            py::print("Exception in session callback:", error.what());
                        }
                    }
                );
            },
            py::arg("callback"),
            R"doc(
            Set the per-step callback, invoked as callback(time, controllers).
        )doc"
        );

    py::class_<
        RS_WorldSessionComponentCamera,
        RS_WorldSessionComponent,
        std::shared_ptr<RS_WorldSessionComponentCamera>>(m, "RS_WorldSessionComponentCamera")
        .def(
            py::init([](std::string name,
                        py::object callback,
                        RVec3rf position,
                        RVec3rf focus,
                        RSize width,
                        RSize height,
                        RSize samples_per_pixel,
                        RSize max_depth,
                        RFloat field_of_view,
                        RFloat defocus_angle,
                        RFloat focus_distance,
                        RFloat min_distance,
                        RFloat max_distance,
                        RFloat zoom_factor) {
                RS_WorldSessionComponentCamera::Callback cpp_callback;
                if (!callback.is_none()) {
                    std::shared_ptr<py::object> py_object(
                        new py::object(std::move(callback)), [](py::object *object) {
                            py::gil_scoped_acquire gil;
                            delete object;
                        }
                    );
                    cpp_callback = [py_object](RBigFloat time) {
                        py::gil_scoped_acquire gil;
                        try {
                            py::cast<py::function> (*py_object)(time);
                        } catch (const py::error_already_set &error) {
                            py::print("Exception in camera callback:", error.what());
                        }
                    };
                }
                return std::make_shared<RS_WorldSessionComponentCamera>(
                    std::move(name),
                    std::move(cpp_callback),
                    position,
                    focus,
                    width,
                    height,
                    samples_per_pixel,
                    max_depth,
                    field_of_view,
                    defocus_angle,
                    focus_distance,
                    min_distance,
                    max_distance,
                    zoom_factor
                );
            }),
            py::arg("name"),
            py::arg("callback") = py::none(),
            py::arg("position") = RVec3rf(0.0, 0.0, 0.4),
            py::arg("focus") = RVec3rf(0.0, 0.0, 0.15),
            py::arg("width") = 1280,
            py::arg("height") = 720,
            py::arg("samples_per_pixel") = 4,
            py::arg("max_depth") = 3,
            py::arg("field_of_view") = 60.0,
            py::arg("defocus_angle") = 0.0,
            py::arg("focus_distance") = 1.0,
            py::arg("min_distance") = 0.05,
            py::arg("max_distance") = RS_MAX_FLOAT,
            py::arg("zoom_factor") = 0.95,
            R"doc(
            Orbit-camera component: injects one free perspective camera and drives its pose.
        )doc"
        )
        .def(
            "move",
            &RS_WorldSessionComponentCamera::move,
            py::arg("orbit_pitch"),
            py::arg("orbit_yaw"),
            py::arg("pan_x"),
            py::arg("pan_y"),
            py::arg("zoom"),
            R"doc(
            Accumulate a mouse gesture; applied on the next control step.
        )doc"
        )
        .def(
            "frame",
            [](const RS_WorldSessionComponentCamera &camera) -> py::object {
                RSize width = 0;
                RSize height = 0;
                std::shared_ptr<const RColor3b[]> rgb = camera.frame(width, height);
                if (!rgb || width == 0 || height == 0) return py::none();
                return py::array_t<uint8_t>(
                    {static_cast<size_t>(height), static_cast<size_t>(width), size_t(3)},
                    reinterpret_cast<const uint8_t *>(rgb.get())
                );
            },
            R"doc(
            The latest RGB frame as a copied (height, width, 3) uint8 array, or None if empty.
        )doc"
        );

#ifdef RISE_ORIGIN_BUILD_CORE_WITH_VR
    py::class_<
        RS_WorldSessionComponentVR,
        RS_WorldSessionComponent,
        std::shared_ptr<RS_WorldSessionComponentVR>>(m, "RS_WorldSessionComponentVR")
        .def(
            py::init([](std::string name,
                        RSize samples_per_pixel,
                        RSize max_depth,
                        RFloat near_depth,
                        RFloat far_depth,
                        RFloat move_speed,
                        RFloat fly_speed,
                        RFloat turn_speed,
                        RFloat stick_dead_zone,
                        RFloat boost,
                        RFloat hud_follow_tau) {
                return std::make_shared<RS_WorldSessionComponentVR>(
                    std::move(name),
                    samples_per_pixel,
                    max_depth,
                    near_depth,
                    far_depth,
                    move_speed,
                    fly_speed,
                    turn_speed,
                    stick_dead_zone,
                    boost,
                    hud_follow_tau
                );
            }),
            py::arg("name"),
            py::arg("samples_per_pixel") = 1,
            py::arg("max_depth") = 3,
            py::arg("near_depth") = 0.05,
            py::arg("far_depth") = 100.0,
            py::arg("move_speed") = 2.0,
            py::arg("fly_speed") = 1.5,
            py::arg("turn_speed") = 90.0,
            py::arg("stick_dead_zone") = 0.2,
            py::arg("boost") = 3.0,
            py::arg("hud_follow_tau") = 0.25,
            R"doc(
            Stereo VR component: injects one stereo camera and presents it to an OpenXR headset.
        )doc"
        )
        .def(
            "setPushForce",
            &RS_WorldSessionComponentVR::setPushForce,
            py::arg("push_force"),
            R"doc(
            Set the newtons applied along the aim ray at a full trigger pull.
        )doc"
        );
#endif  // RISE_ORIGIN_BUILD_CORE_WITH_VR

#ifdef RISE_ORIGIN_BUILD_CORE_WITH_MCP
    py::class_<
        RS_WorldSessionComponentMCP,
        RS_WorldSessionComponent,
        std::shared_ptr<RS_WorldSessionComponentMCP>>(m, "RS_WorldSessionComponentMCP")
        .def(
            py::init([](std::string name,
                        std::string host,
                        int port,
                        std::string token,
                        bool is_python_exec_enabled) {
                return std::make_shared<RS_WorldSessionComponentMCP>(
                    std::move(name), std::move(host), port, std::move(token), is_python_exec_enabled
                );
            }),
            py::arg("name"),
            py::arg("host") = "127.0.0.1",
            py::arg("port") = 8765,
            py::arg("token") = "",
            py::arg("is_python_exec_enabled") = false,
            R"doc(
            MCP server component: hosts a Model Context Protocol server so an AI client can read,
            change, see, and be embodied in the world.
        )doc"
        )
        .def(
            "register_tool",
            [](RS_WorldSessionComponentMCP &component,
               std::string name,
               std::string description,
               std::string input_schema,
               py::object handler) {
                std::shared_ptr<py::object> py_handler(
                    new py::object(std::move(handler)), [](py::object *object) {
                        py::gil_scoped_acquire gil;
                        delete object;
                    }
                );
                RS_MCPToolHandler cpp_handler = [py_handler](
                                                    std::shared_ptr<RS_WorldSessionHandle> handle,
                                                    std::string arguments
                                                ) -> std::string {
                    py::gil_scoped_acquire gil;
                    try {
                        py::object result = py::cast<py::function>(*py_handler)(
                            RSPY_WorldHandle(std::move(handle)), std::move(arguments)
                        );
                        return py::cast<std::string>(result);
                    } catch (const py::error_already_set &error) {
                        throw std::runtime_error(error.what());
                    }
                };
                component.registerTool(
                    std::move(name),
                    std::move(description),
                    std::move(input_schema),
                    std::move(cpp_handler)
                );
            },
            py::arg("name"),
            py::arg("description"),
            py::arg("input_schema"),
            py::arg("handler"),
            R"doc(
            Register a Python authored tool served next to the built in ones. The handler takes the
            full world handle and the arguments json string, and returns the result json string.
        )doc"
        )
        .def(
            "unregister_tool",
            &RS_WorldSessionComponentMCP::unregisterTool,
            py::arg("name"),
            R"doc(
            Remove a registered tool by name.
        )doc"
        )
        .def(
            "list_registered_tools",
            &RS_WorldSessionComponentMCP::getRegisteredToolNames,
            R"doc(
            Get the names of every registered tool.
        )doc"
        );
#endif  // RISE_ORIGIN_BUILD_CORE_WITH_MCP
}
