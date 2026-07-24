# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the module protocol: the decorator, the registry, and help."""

import time
import warnings

import pytest

from rise_origin.create.data import ModuleReference
from rise_origin.create.protocol import (
    Float,
    ModuleError,
    help_module,
    list_modules,
    module,
    settings,
)
from rise_origin.create.protocol.module import ModuleRegistry, ModuleSpec, registry


def make_spec(name, version, dependencies=(), deprecation=None):
    def body(editor_access, values):
        """A graph test body."""

    return ModuleSpec(
        name=name,
        version=version,
        schema=settings(),
        dependencies=tuple(dependencies),
        tags=(),
        doc="",
        fn=body,
        deprecation=deprecation,
    )


class TestModuleDecorator:
    def test_registration_reaches_the_registry(self):
        @module(name="rise_origin.test.proto.stone", version="1.0.0")
        def stone(editor_access, values):
            """A stone."""

        spec = registry.resolve(
            ModuleReference.from_text("rise_origin.test.proto.stone")
        )
        assert spec.version == "1.0.0"
        assert spec.doc.strip() == "A stone."
        assert stone.spec is spec

    def test_three_level_names_are_refused(self):
        with pytest.raises(ModuleError):
            module(name="test.proto.pebble", version="1.0.0")

    def test_versions_must_be_semver(self):
        with pytest.raises(ModuleError):
            module(name="rise_origin.test.proto.pebble", version="1.0")

    def test_direct_calls_point_at_the_editor(self):
        @module(name="rise_origin.test.proto.brick", version="1.0.0")
        def brick(editor_access, values):
            """A brick."""

        with pytest.raises(ModuleError):
            brick()


class TestModuleRegistry:
    def test_the_newest_matching_version_wins(self):
        name = "rise_origin.test.proto.plank"
        for version in ("1.0.0", "1.2.0", "2.0.0"):

            @module(name=name, version=version)
            def plank(editor_access, values):
                """A plank."""

        assert registry.resolve(ModuleReference.from_text(name)).version == "2.0.0"
        assert (
            registry.resolve(ModuleReference.from_text(f"{name}@^1")).version == "1.2.0"
        )
        assert (
            registry.resolve(ModuleReference.from_text(f"{name}@1.0.0")).version
            == "1.0.0"
        )

    def test_unknown_names_and_versions_raise_key_errors(self):
        with pytest.raises(KeyError):
            registry.resolve(
                ModuleReference.from_text("rise_origin.test.proto.unknown")
            )
        name = "rise_origin.test.proto.beam"

        @module(name=name, version="1.0.0")
        def beam(editor_access, values):
            """A beam."""

        with pytest.raises(KeyError):
            registry.resolve(ModuleReference.from_text(f"{name}@^9"))

    def test_list_modules_sorts_the_names(self):
        @module(name="rise_origin.test.proto.tile", version="1.0.0")
        def tile(editor_access, values):
            """A tile."""

        names = list_modules()
        assert "rise_origin.test.proto.tile" in names
        assert names == sorted(names)

    def test_help_carries_the_full_manual(self):
        name = "rise_origin.test.proto.slab"

        @module(
            name=name,
            version="1.3.0",
            settings=settings(width=Float(0.1, 2.0, default=0.5)),
            dependencies=("rise_origin.test.proto.tile",),
            tags=("flat",),
        )
        def slab(editor_access, values):
            """A slab of stone."""

        manual = help_module(name)
        assert f"{name}@1.3.0" in manual
        assert "A slab of stone." in manual
        assert "rise_origin.test.proto.tile" in manual
        assert "flat" in manual
        assert "width: float in [0.1, 2]" in manual
        assert "seed: int" in manual
        assert "not deterministic" not in manual

    def test_help_shows_a_non_deterministic_declaration(self):
        name = "rise_origin.test.proto.scree"

        @module(name=name, version="1.0.0", is_deterministic=False)
        def scree(editor_access, values):
            """A simulated stone pile."""

        assert not scree.spec.is_deterministic
        manual = help_module(name)
        assert "not deterministic: the same settings may build different content" in (
            manual
        )


class TestModuleGraph:
    def test_register_refuses_a_malformed_dependency(self):
        fresh = ModuleRegistry()
        with pytest.raises(ModuleError):
            fresh.register(
                make_spec("rise_origin.test.graph.bad", "1.0.0", ["not a reference"])
            )

    def test_validate_reports_every_problem_together(self):
        fresh = ModuleRegistry()
        fresh.register(
            make_spec(
                "rise_origin.test.graph.a",
                "1.0.0",
                [
                    "rise_origin.test.graph.void",
                    "rise_origin.test.graph.b@^9",
                    "rise_origin.test.graph.b",
                ],
            )
        )
        fresh.register(
            make_spec("rise_origin.test.graph.b", "1.0.0", ["rise_origin.test.graph.a"])
        )
        with pytest.raises(ModuleError) as caught:
            fresh.validate()
        message = str(caught.value)
        assert "rise_origin.test.graph.void" in message
        assert "rise_origin.test.graph.b@^9" in message
        assert "cycle" in message

    def test_resolve_pins_backtracks_to_a_satisfiable_version(self):
        fresh = ModuleRegistry()
        clay = "rise_origin.test.graph.clay"
        pad = "rise_origin.test.graph.pad"
        fresh.register(make_spec(clay, "1.4.0"))
        fresh.register(make_spec(clay, "2.1.0"))
        fresh.register(make_spec(pad, "1.0.0", [f"{clay}@^1"]))
        fresh.register(make_spec(pad, "2.0.0", [f"{clay}@^2"]))
        pins = fresh.resolve_pins({clay: ["^1"], pad: [None]})
        assert pins[pad].version == "1.0.0"
        assert pins[clay].version == "1.4.0"

    def test_resolve_pins_lists_unresolved_names_together(self):
        fresh = ModuleRegistry()
        fresh.register(make_spec("rise_origin.test.graph.solo", "1.0.0"))
        with pytest.raises(ModuleError) as caught:
            fresh.resolve_pins(
                {
                    "rise_origin.test.graph.ghost": [None],
                    "rise_origin.test.graph.solo": ["^9"],
                }
            )
        message = str(caught.value)
        assert "rise_origin.test.graph.ghost" in message
        assert "^9" in message

    def test_resolving_a_deprecated_version_warns(self):
        fresh = ModuleRegistry()
        name = "rise_origin.test.graph.retired"
        fresh.register(make_spec(name, "1.0.0", deprecation="use ^2 instead"))
        fresh.register(make_spec(name, "2.0.0"))
        with pytest.warns(DeprecationWarning, match="use \\^2 instead"):
            spec = fresh.resolve(ModuleReference.from_text(f"{name}@^1"))
        assert spec.version == "1.0.0"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fresh.resolve(ModuleReference.from_text(name)).version == "2.0.0"

    def test_pinning_a_deprecated_version_names_the_dependents(self):
        fresh = ModuleRegistry()
        old = "rise_origin.test.graph.old_way"
        user = "rise_origin.test.graph.user"
        fresh.register(make_spec(old, "1.0.0", deprecation="grown differently now"))
        fresh.register(make_spec(user, "1.0.0", [f"{old}@^1"]))
        with pytest.warns(DeprecationWarning) as caught:
            fresh.resolve_pins({user: [None]})
        messages = [str(warning.message) for warning in caught]
        assert any(
            "grown differently now" in message and f"{user}@1.0.0" in message
            for message in messages
        )

    def test_help_shows_the_deprecation(self):
        name = "rise_origin.test.proto.sunset"

        @module(name=name, version="1.0.0", deprecation="ends with the season")
        def sunset(editor_access, values):
            """A sunset."""

        with pytest.warns(DeprecationWarning):
            manual = help_module(name)
        assert "DEPRECATED: ends with the season" in manual

    def test_resolve_pins_scales_to_thousands_of_versions(self):
        fresh = ModuleRegistry()
        names = [f"rise_origin.test.graph.kind_{index:03d}" for index in range(400)]
        for position, name in enumerate(names):
            follower = names[position + 1] if position + 1 < len(names) else None
            for major in range(1, 6):
                dependencies = [f"{follower}@^{major}"] if follower else []
                fresh.register(make_spec(name, f"{major}.0.0", dependencies))
        constraints = {name: [None] for name in names[:300]}
        constraints[names[0]] = ["^3"]
        started = time.monotonic()
        pins = fresh.resolve_pins(constraints)
        elapsed = time.monotonic() - started
        assert len(pins) == len(names)
        assert all(pins[name].version == "3.0.0" for name in names)
        assert elapsed < 2.0
