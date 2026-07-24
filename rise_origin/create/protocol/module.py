# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The module protocol: one decorator and the registry behind editor.create.

A generation module is opaque Python behind one fixed signature, declared
like this::

    @cr.module(
        name="rise_origin.nature.tree.maple",
        version="1.2.0",
        settings=cr.settings(height=cr.Float(2.0, 12.0, default=6.0)),
        dependencies=("rise_origin.material.nature.plant@^1",),
    )
    def maple(editor_access, settings):
        \"\"\"A maple tree.\"\"\"
        rng = cr.Rng(settings.seed)  # Note: it's settings default seed field
        return cr.Draft(editor_access, "maple", ...)

Required per module:

1. The body is pure. All randomness flows from ``settings.seed`` through
   an ``Rng``; no clock, no reads outside the handed-in access.
2. Building goes only through ``editor.create``. Calling ``maple(...)``
   directly refuses and points the caller at an editor.
3. Old versions stay in the package. Several versions of one name
   register side by side, each with its own body, settings, and
   dependencies; that is how a mod keeps old worlds rebuildable. To
   retire one, mark it with ``deprecation=`` first and remove it only a
   few releases later, once users had time to move.

How dependencies resolve:

1. Registration is order free. Mods import in any order and nothing
   resolves at register time. Two mods must not register the same name;
   the organization level of the name is the ownership boundary. Patching
   or overriding another mod's module is not allowed: depend on it and
   compose, or release a new version.
2. ``validate`` checks the whole graph once per process. Every problem
   raises together, the way a game refuses a broken mod set at load.
3. ``resolve_pins`` pins one version per module for a world, the way pip
   pins one version per package. Dependencies of dependencies join the
   pin set automatically.
"""

import functools
import re
import warnings
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from resolvelib import AbstractProvider, BaseReporter, ResolutionImpossible, Resolver

from ..constants import RESOLVE_MAX_ROUNDS
from ..data.module_reference import ModuleReference
from .types import SettingsSchema

__all__ = [
    "ModuleError",
    "ModuleSpec",
    "ModuleRegistry",
    "registry",
    "module",
    "list_modules",
    "help_module",
]

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){3}$")
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class ModuleError(RuntimeError):
    """A module declaration or call that breaks the protocol."""


@dataclass(frozen=True)
class ModuleSpec:
    """One registered module: its contract and its opaque body.

    Attributes:
        name: Full four-level module name, organization included.
        version: Concrete semver version of this registration.
        schema: The declared settings, automatic seed included.
        dependencies: Module names this one needs built first.
        tags: Loose search strings.
        doc: The usage text shown by help.
        fn: The opaque body with signature
            ``fn(editor_access, settings) -> Draft``.
        deprecation: The message warned whenever this version resolves;
            None means not deprecated.
        is_deterministic: Whether the same identity always builds the
            same bytes; a simulation-based body declares False and the
            journal records its built content instead of the recipe.
    """

    name: str
    version: str
    schema: SettingsSchema
    dependencies: Tuple[str, ...]
    tags: Tuple[str, ...]
    doc: str
    fn: Callable
    deprecation: Optional[str] = None
    is_deterministic: bool = True


class ModuleRegistryProvider(AbstractProvider):
    """The registry's translator to resolvelib's backtracking resolver.

    Requirements are module references and candidates are module specs;
    the provider only maps between them and the registry's version tables.
    Candidates come newest first, so the resolver prefers the newest
    version satisfying every requirement and backtracks to older ones only
    when a dependency contradicts the choice.
    """

    def __init__(self, registry: "ModuleRegistry") -> None:
        self._registry = registry

    def identify(
        self, requirement_or_candidate: Union[ModuleReference, ModuleSpec]
    ) -> str:
        """Return the module name a requirement or candidate is about.

        Args:
            requirement_or_candidate: A module reference or a module spec.
        """
        if isinstance(requirement_or_candidate, ModuleSpec):
            return requirement_or_candidate.name
        return requirement_or_candidate.get_name()

    def get_preference(
        self, identifier, resolutions, candidates, information, backtrack_causes
    ) -> str:
        """Return a module's resolution order key; names sort alphabetically.

        Args:
            identifier: The module name to rank.
            resolutions: The versions pinned so far, per module name.
            candidates: The open candidates, per module name.
            information: The requirements and their sources, per module name.
            backtrack_causes: The requirements behind the latest backtrack.
        """
        return identifier

    def find_matches(self, identifier, requirements, incompatibilities) -> List:
        """Return a name's specs satisfying every requirement, newest first.

        Args:
            identifier: The module name matched.
            requirements: The requirements to satisfy together, per module
                name.
            incompatibilities: The specs already refused, per module name.
        """
        matching = None
        for requirement in requirements[identifier]:
            versions = self._registry._get_matching_versions(
                identifier, requirement.version_text
            )
            matching = versions if matching is None else matching & versions
        refused = {spec.version for spec in incompatibilities[identifier]}
        if not matching:
            return []
        return [
            self._registry._entries[identifier][version]
            for version in self._registry._versions_newest_first.get(identifier, ())
            if version in matching and version not in refused
        ]

    def is_satisfied_by(self, requirement, candidate) -> bool:
        """Return whether a spec's version satisfies a reference's version part.

        Args:
            requirement: The module reference stating the version part.
            candidate: The module spec tested.
        """
        return candidate.version in self._registry._get_matching_versions(
            candidate.name, requirement.version_text
        )

    def get_dependencies(self, candidate) -> List[ModuleReference]:
        """Return a spec's dependency declarations as module references.

        Args:
            candidate: The module spec whose dependencies open.
        """
        return [ModuleReference.from_text(text) for text in candidate.dependencies]


class ModuleRegistry:
    """The registry keeping track of every create module, held as the
    process singleton ``registry``."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, ModuleSpec]] = {}
        self._generation = 0
        self._validate_generation: Optional[int] = None
        self._tables_generation: Optional[int] = None
        self._versions_newest_first: Dict[str, List[str]] = {}
        self._dependency_names: Dict[str, Set[str]] = {}
        self._topological_names: List[str] = []
        self._match_cache: Dict[Tuple[str, Optional[str]], Set[str]] = {}
        self._pins_cache: Dict[tuple, Dict[str, ModuleSpec]] = {}

    def register(self, spec: ModuleSpec) -> None:
        """Add one module version; the same version registers over itself.

        Registration is order free: nothing resolves here, so mods import
        in any order. Registering the same name and version again replaces
        the stored body, so re-running a cell on the editor kernel
        notebook while writing a module just works.

        Args:
            spec: The module spec to register.

        Raises:
            ModuleError: A dependency text is not a module reference.
        """
        for text in spec.dependencies:
            try:
                ModuleReference.from_text(text)
            except ValueError as error:
                raise ModuleError(
                    f"module '{spec.name}' declares dependency '{text}' "
                    "which is not a module reference"
                ) from error
        self._entries.setdefault(spec.name, {})[spec.version] = spec
        self._generation += 1

    def validate(self) -> None:
        """Validate the whole module graph; every problem raises together.

        Validation covers every registered module: each dependency names
        a registered module, each dependency version part matches at least
        one registered version, and no dependency cycle runs between
        module names. The result caches until a new registration, so later
        calls cost one integer compare; ``resolve_pins`` runs it
        automatically, and a mod loader may call it directly at startup.

        Raises:
            ModuleError: The report of every problem in the graph.
        """
        if self._validate_generation == self._generation:
            return
        import rise_origin.create.modules  # noqa: F401 loads first-party kinds

        self._refresh_version_tables()
        problems = []
        graph: Dict[str, Set[str]] = {name: set() for name in self._entries}
        for name, versions in sorted(self._entries.items()):
            for version, spec in sorted(versions.items()):
                for text in spec.dependencies:
                    reference = ModuleReference.from_text(text)
                    dependency = reference.get_name()
                    graph[name].add(dependency)
                    if dependency not in self._entries:
                        problems.append(
                            f"{name}@{version} depends on '{dependency}' "
                            "which is not registered"
                        )
                    elif not self._get_matching_versions(
                        dependency, reference.version_text
                    ):
                        problems.append(
                            f"{name}@{version} depends on '{text}' which no "
                            "registered version matches"
                        )
        self._dependency_names = graph
        try:
            self._topological_names = list(TopologicalSorter(graph).static_order())
        except CycleError as error:
            problems.append("dependency cycle: " + " -> ".join(error.args[1]))
        if problems:
            raise ModuleError(
                "the module graph has problems:\n- " + "\n- ".join(problems)
            )
        self._validate_generation = self._generation

    def resolve(self, reference: ModuleReference) -> ModuleSpec:
        """Return the newest registered version a reference accepts.

        The reference's version part is a specifier; no version part and
        ``*`` accept everything. Resolving a deprecated version warns
        with its deprecation message; the choice itself never changes.

        Args:
            reference: The module reference to resolve, version part
                included.

        Raises:
            KeyError: No module carries the name, or no registered version
                satisfies the specifier.
        """
        name = reference.get_name()
        versions = self._entries.get(name)
        if not versions:
            import rise_origin.create.modules  # noqa: F401 loads first-party kinds

            versions = self._entries.get(name)
        if not versions:
            raise KeyError(f"no module named '{name}'")
        matching = [
            version
            for version in versions
            if reference.is_version_matching(ModuleReference.parse_version(version))
        ]
        if not matching:
            raise KeyError(f"no version of '{name}' matches '{reference.version_text}'")
        best = max(matching, key=ModuleReference.parse_version)
        spec = versions[best]
        if spec.deprecation is not None:
            warnings.warn(
                f"module '{spec.name}@{spec.version}' is deprecated: "
                f"{spec.deprecation}",
                DeprecationWarning,
                stacklevel=2,
            )
        return spec

    def resolve_pins(
        self, constraints: Dict[str, Sequence[Optional[str]]]
    ) -> Dict[str, ModuleSpec]:
        """Return one pinned spec per module a constraint set reaches.

        Resolution is pip's model: every constrained module and everything
        reachable through dependency declarations pins to one version
        satisfying every version text stated for it, preferring the
        newest. A conflict-free graph, the common case, pins in one direct
        pass over the edges; only a genuine version conflict pays the
        backtracking search, which tries older candidates until a version
        set fits together. Results memoize per constraint set until a new
        registration, and ``validate`` runs first, so a broken graph
        raises before any resolution. Pinning a deprecated version warns
        with its message and names who required it; the pin itself never
        changes.

        Args:
            constraints: The version texts per module name, gathered from
                one world's requires and placed identities; None and ``*``
                accept everything.

        Raises:
            ModuleError: A constrained name is not registered, a version
                text matches nothing, or no version set satisfies every
                constraint together; the first two list together.
        """
        self.validate()
        key = tuple(
            (name, ModuleRegistry._make_unique_texts(constraints[name]))
            for name in sorted(constraints)
        )
        held = self._pins_cache.get(key)
        if held is not None:
            return dict(held)
        problems = []
        for name, texts in key:
            if name not in self._entries:
                problems.append(f"no module named '{name}'")
                continue
            for text in texts:
                if not self._get_matching_versions(name, text):
                    problems.append(f"no version of '{name}' matches '{text or '*'}'")
        if problems:
            raise ModuleError("unresolved modules:\n- " + "\n- ".join(problems))
        pins = self._resolve_pins_directly(key)
        if pins is None:
            pins = self._resolve_pins_by_backtracking(key)
        pins = {name: pins[name] for name in sorted(pins)}
        ModuleRegistry._warn_deprecated_pins(key, pins)
        self._pins_cache[key] = pins
        return dict(pins)

    def get_matching_versions(self, name: str, version_text: Optional[str]) -> Set[str]:
        """Return the registered versions of a name a version text accepts.

        None and ``*`` accept everything; an unregistered name answers
        the empty set. Answers cache until a new registration.

        Args:
            name: The full four-level module name looked up.
            version_text: The version text tested against each registered
                version, a specifier or a concrete version; None accepts
                everything.
        """
        self._refresh_version_tables()
        return self._get_matching_versions(name, version_text)

    def list(self) -> List[str]:
        """Return every registered module name, sorted."""
        import rise_origin.create.modules  # noqa: F401 loads first-party kinds

        return sorted(self._entries)

    def help(self, name: str) -> str:
        """Return the usage manual of a module's newest matching version.

        The manual carries the resolved version, the usage text, the
        dependencies and tags, and every setting with its type, range, and
        default.

        Args:
            name: The module name to look up, with an optional version part.
        """
        spec = self.resolve(ModuleReference.from_text(name))
        lines = [f"{spec.name}@{spec.version}"]
        if spec.deprecation is not None:
            lines.append(f"DEPRECATED: {spec.deprecation}")
        if not spec.is_deterministic:
            lines.append(
                "not deterministic: the same settings may build different content"
            )
        if spec.doc:
            lines.append(spec.doc.strip())
        if spec.dependencies:
            lines.append("dependencies: " + ", ".join(spec.dependencies))
        if spec.tags:
            lines.append("tags: " + ", ".join(spec.tags))
        lines.append("settings:")
        for field_name, field_json in spec.schema.to_json().items():
            lines.append("  " + ModuleRegistry._describe_field(field_name, field_json))
        return "\n".join(lines)

    def _resolve_pins_directly(
        self, key: Tuple[Tuple[str, Tuple[Optional[str], ...]], ...]
    ) -> Optional[Dict[str, ModuleSpec]]:
        """Pin without backtracking; None when only backtracking can answer.

        Names walk dependents before dependencies, an order the validated
        acyclic graph guarantees, so every version text is known before
        its name pins and one pass over the edges decides everything. A
        name no chosen version references gathers no texts and stays out.
        An empty intersection anywhere abandons the pass, because an older
        dependent candidate may still fit.
        """
        texts_by_name: Dict[str, List[Optional[str]]] = {
            name: list(texts) for name, texts in key
        }
        closure = set()
        stack = list(texts_by_name)
        while stack:
            name = stack.pop()
            if name in closure:
                continue
            closure.add(name)
            stack.extend(self._dependency_names.get(name, ()))
        chosen: Dict[str, ModuleSpec] = {}
        for name in reversed(self._topological_names):
            if name not in closure:
                continue
            texts = texts_by_name.get(name)
            if not texts:
                continue
            matching = None
            for text in set(texts):
                versions = self._get_matching_versions(name, text)
                matching = versions if matching is None else matching & versions
            if not matching:
                return None
            newest = next(
                version
                for version in self._versions_newest_first[name]
                if version in matching
            )
            spec = self._entries[name][newest]
            chosen[name] = spec
            for text in spec.dependencies:
                reference = ModuleReference.from_text(text)
                texts_by_name.setdefault(reference.get_name(), []).append(
                    reference.version_text
                )
        return chosen

    def _resolve_pins_by_backtracking(
        self, key: Tuple[Tuple[str, Tuple[Optional[str], ...]], ...]
    ) -> Dict[str, ModuleSpec]:
        """Pin through resolvelib's backtracking search.

        Raises:
            ModuleError: No version set satisfies every constraint
                together; the message lists the requirements that cannot
                coexist and who needs them.
        """
        requirements = [
            ModuleReference.from_text(name if text is None else f"{name}@{text}")
            for name, texts in key
            for text in texts
        ]
        resolver = Resolver(ModuleRegistryProvider(self), BaseReporter())
        try:
            result = resolver.resolve(requirements, max_rounds=RESOLVE_MAX_ROUNDS)
        except ResolutionImpossible as error:
            listing = "; ".join(
                cause.requirement.to_text()
                + (
                    f" (needed by {cause.parent.name}@{cause.parent.version})"
                    if cause.parent is not None
                    else ""
                )
                for cause in error.causes
            )
            raise ModuleError(f"no version set satisfies: {listing}") from error
        return dict(result.mapping)

    def _get_matching_versions(
        self, name: str, version_text: Optional[str]
    ) -> Set[str]:
        """Answer a match lookup assuming the version tables are fresh."""
        key = (name, version_text)
        held = self._match_cache.get(key)
        if held is None:
            reference = ModuleReference.from_text(
                name if version_text is None else f"{name}@{version_text}"
            )
            held = {
                version
                for version in self._versions_newest_first.get(name, ())
                if reference.is_version_matching(ModuleReference.parse_version(version))
            }
            self._match_cache[key] = held
        return held

    def _refresh_version_tables(self) -> None:
        """Rebuild the version lookup tables after a new registration."""
        if self._tables_generation == self._generation:
            return
        self._versions_newest_first = {
            name: sorted(versions, key=ModuleReference.parse_version, reverse=True)
            for name, versions in self._entries.items()
        }
        self._match_cache.clear()
        self._pins_cache.clear()
        self._tables_generation = self._generation

    @staticmethod
    def _warn_deprecated_pins(
        key: Tuple[Tuple[str, Tuple[Optional[str], ...]], ...],
        pins: Dict[str, ModuleSpec],
    ) -> None:
        """Warn for every deprecated pin, naming who required it."""
        constrained = {name for name, _texts in key}
        for name, spec in pins.items():
            if spec.deprecation is None:
                continue
            dependents = [
                f"{holder.name}@{holder.version}"
                for holder in pins.values()
                if any(
                    ModuleReference.from_text(text).get_name() == name
                    for text in holder.dependencies
                )
            ]
            if name in constrained:
                dependents.append("the world")
            warnings.warn(
                f"'{spec.name}@{spec.version}' is deprecated: "
                f"{spec.deprecation}; required by " + ", ".join(dependents),
                DeprecationWarning,
                stacklevel=3,
            )

    @staticmethod
    def _make_unique_texts(texts: Sequence[Optional[str]]) -> Tuple[Optional[str], ...]:
        """Return a constraint's version texts deduplicated and ordered."""
        return tuple(sorted(set(texts), key=lambda text: (text is None, text or "")))

    @staticmethod
    def _describe_field(name: str, field_json: dict) -> str:
        """Return one settings line of the help manual."""
        kind = field_json["type"]
        if kind in ("float", "int"):
            return (
                f"{name}: {kind} in [{field_json['min']:g}, "
                f"{field_json['max']:g}], default {field_json['default']:g}"
            )
        if kind == "float_list":
            length = field_json["length"]
            count = "any number of" if length is None else f"exactly {length}"
            return (
                f"{name}: {count} floats in [{field_json['min']:g}, "
                f"{field_json['max']:g}], default {field_json['default']}"
            )
        if kind in ("choice", "voxel_size"):
            options = ", ".join(field_json["options"])
            return f"{name}: one of {options}, default {field_json['default']}"
        return f"{name}: {kind}, default {field_json['default']!r}"


registry = ModuleRegistry()


def module(
    *,
    name: str,
    version: str,
    settings: Optional[SettingsSchema] = None,
    dependencies: Tuple[str, ...] = (),
    tags: Tuple[str, ...] = (),
    doc: Optional[str] = None,
    deprecation: Optional[str] = None,
    is_deterministic: bool = True,
) -> Callable:
    """Declare the decorated function as a create module.

    The body has signature ``fn(editor_access, settings) -> Draft`` and
    must be pure: all randomness from ``settings.seed``, no clock, no reads
    outside the access. Decorating registers the module; building goes only
    through ``editor.create``, so calling the decorated function directly
    raises. The registered spec stays reachable as its ``spec`` attribute.

    Args:
        name: Full four-level module name, such as
            ``rise_origin.nature.tree.maple``.
        version: Concrete semver; any output change is a new version.
        settings: The declared settings schema; None declares none, the
            automatic seed still applies.
        dependencies: Module names this one needs built first, material
            modules above all.
        tags: Loose search strings.
        doc: Usage text for help; None takes the function docstring.
        deprecation: The message warned whenever this version resolves;
            None means not deprecated. Deprecate an old version first and
            remove it only a few releases later.
        is_deterministic: Whether the same identity always builds the
            same bytes; a simulation-based body declares False, its cache
            entry becomes a first-build snapshot, and sessions record its
            built content in the journal.
    """
    if not _NAME_PATTERN.match(name):
        raise ModuleError(
            f"module name '{name}' must be four dotted lowercase levels, "
            "organization.area.group.kind"
        )
    if not _VERSION_PATTERN.match(version):
        raise ModuleError(f"module version '{version}' must be semver")
    schema = settings if settings is not None else SettingsSchema()

    def decorate(fn: Callable) -> Callable:
        spec = ModuleSpec(
            name=name,
            version=version,
            schema=schema,
            dependencies=tuple(dependencies),
            tags=tuple(tags),
            doc=doc if doc is not None else (fn.__doc__ or ""),
            fn=fn,
            deprecation=deprecation,
            is_deterministic=is_deterministic,
        )
        registry.register(spec)

        @functools.wraps(fn)
        def refuse(*args: object, **kwargs: object) -> None:
            raise ModuleError(
                f"'{name}' builds only through an editor; call "
                f"editor.create('{name}', settings)"
            )

        refuse.spec = spec
        return refuse

    return decorate


list_modules = registry.list
help_module = registry.help
