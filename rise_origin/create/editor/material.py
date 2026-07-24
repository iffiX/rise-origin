# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The editor's material tables: declared requires and gathered data.

A material definition is nothing but plain rsc essences carried in a
draft's rscs; there is no wrapper layer over them. The two tables here
manage the material data the editor works over: the require table holds
what the session declared it will use, so the simulation step can freeze
over it; the data table holds the named essences and resources one compile
gathers, so emission and the embedding gate can answer from it. Execution
lives on the editor; the tables only register, look up, and answer.

A material module is a package releasing several materials; each released
``Material`` essence carries the module name joined to the material's name
with a colon, such as ``rise_origin.material.nature.plant:leaf``, and
structures reference materials by exactly that string; that one convention
is what lets a require, a module, and a voxel's material reference meet.
"""

import hashlib
import os
import tempfile
from typing import Dict, List, Sequence, Set, Tuple

from rise_origin import (
    RSC_Material,
    RSC_MaterialPairPropertyBase,
    RSC_MaterialPhysicalPropertyFixed,
    RSC_MaterialPhysicalPropertyRigid,
    RSC_MaterialPhysicalPropertySoft,
)

from ..data.material import MaterialRequire
from ..data.module_reference import ModuleReference

__all__ = [
    "EditorMaterialError",
    "EditorMaterialRequireTable",
    "EditorMaterialDataTable",
]


class EditorMaterialError(RuntimeError):
    """A material declaration or resolution that cannot be satisfied."""


class EditorMaterialRequireTable:
    """The material requires the session declared.

    The table holds one merged require per module reference and material,
    version part included, keyed in declaration order. Declarations add
    freely until the table freezes with the simulation, so a frozen
    session can hold any new content against a fixed material set.

    Args:
        requires: Requires declared at construction.
    """

    def __init__(self, requires: Sequence[MaterialRequire] = ()) -> None:
        self._is_frozen = False
        self._require_map: Dict[Tuple[ModuleReference, str], MaterialRequire] = {}
        self.add_requires(requires)

    @property
    def is_frozen(self) -> bool:
        """Whether the table refuses new declarations."""
        return self._is_frozen

    def freeze(self) -> None:
        """Lock the table so new declarations are refused."""
        self._is_frozen = True

    def unfreeze(self) -> None:
        """Open the table so declarations add again."""
        self._is_frozen = False

    def add_requires(self, requires: Sequence[MaterialRequire]) -> None:
        """Add declared requires; refused once the table is frozen.

        A require whose module reference and material are already
        declared, version part included, merges its voxel sizes into the
        held entry; different version parts of one material stay separate
        entries.

        Args:
            requires: The requires to declare.
        """
        if self._is_frozen:
            raise EditorMaterialError(
                "the material table is frozen with the simulation; declare "
                "requires before freezing"
            )
        for require in requires:
            held_key = (require.module_reference, require.material_reference)
            held = self._require_map.get(held_key)
            self._require_map[held_key] = (
                require if held is None else held.merge(require)
            )

    def get_requires(self) -> List[MaterialRequire]:
        """Return every declared require, in declaration order."""
        return list(self._require_map.values())

    def has_require(self, require: MaterialRequire) -> bool:
        """Return whether a declared require covers the given require.

        Covered means one declared entry names the same module and
        material, version parts aside, and carries every one of the given
        require's voxel sizes, compared rounded.

        Args:
            require: The require to look up.
        """
        return any(
            entry.is_superset_of(require) for entry in self._require_map.values()
        )

    def get_required_material_name_and_voxel_sizes(self) -> Dict[str, Set[float]]:
        """Return every declared world material name with its voxel sizes.

        Names are the colon-joined references the compiled world uses;
        sizes union across the version parts of one material and come
        rounded for comparison across float32.
        """
        table: Dict[str, Set[float]] = {}
        for entry in self._require_map.values():
            sizes = table.setdefault(entry.get_world_material_reference(), set())
            for size in entry.voxel_size:
                sizes.add(MaterialRequire.round_voxel_size(size))
        return table


class EditorMaterialDataTable:
    """The named essences and resources one compile gathers.

    Registration is checked by content: the same name registering
    identical bytes merges silently, and different bytes are refused,
    which is what makes name lookups safe. The table answers what compile
    asks: whether a material is present, what physical kinds each
    material carries, and the essence list in emission order.
    """

    def __init__(self) -> None:
        self._essences_by_name: Dict[str, object] = {}
        self._resources_by_name: Dict[str, object] = {}

    def add_essence(self, essence: object) -> None:
        """Register one essence by its name.

        Args:
            essence: The essence to register; a name collision with
                different content is refused.
        """
        EditorMaterialDataTable._add_rsc(self._essences_by_name, essence)

    def add_resource(self, resource: object) -> None:
        """Register one resource by its name.

        Args:
            resource: The resource to register; a name collision with
                different content is refused.
        """
        EditorMaterialDataTable._add_rsc(self._resources_by_name, resource)

    def has_material(self, name: str) -> bool:
        """Return whether a Material essence with the name is registered.

        Args:
            name: The material name looked up.
        """
        return isinstance(self._essences_by_name.get(name), RSC_Material)

    def get_material_physical_kinds(self) -> Dict[str, Set[str]]:
        """Return every registered material's physical kinds.

        Kinds read from the registered property essences: fixed, rigid,
        and soft; a material with no physical property answers an empty
        set.
        """
        kinds: Dict[str, Set[str]] = {}
        for essence in self._essences_by_name.values():
            if isinstance(essence, RSC_MaterialPhysicalPropertyFixed):
                kinds.setdefault(essence.material_reference, set()).add("fixed")
            elif isinstance(essence, RSC_MaterialPhysicalPropertyRigid):
                kinds.setdefault(essence.material_reference, set()).add("rigid")
            elif isinstance(essence, RSC_MaterialPhysicalPropertySoft):
                kinds.setdefault(essence.material_reference, set()).add("soft")
            elif isinstance(essence, RSC_Material):
                kinds.setdefault(essence.name, set())
        return kinds

    def get_essences(self) -> List[object]:
        """Return every registered essence, ordered for emission.

        Materials come first, single-material properties second, and pair
        properties last, so every reference points backward at load; a
        pair property is kept only while both its referenced materials are
        registered. Registration order is kept inside each group.
        """
        material_names = {
            name
            for name, essence in self._essences_by_name.items()
            if isinstance(essence, RSC_Material)
        }
        ordered = []
        for essence in self._essences_by_name.values():
            if isinstance(essence, RSC_Material):
                ordered.append((0, essence))
            elif isinstance(essence, RSC_MaterialPairPropertyBase):
                if (
                    essence.first_material_reference in material_names
                    and essence.second_material_reference in material_names
                ):
                    ordered.append((2, essence))
            else:
                ordered.append((1, essence))
        ordered.sort(key=lambda item: item[0])
        return [essence for _rank, essence in ordered]

    def get_resources(self) -> List[object]:
        """Return every registered resource, in registration order."""
        return list(self._resources_by_name.values())

    @staticmethod
    def _add_rsc(table: Dict[str, object], rsc: object) -> None:
        """Register one named rsc, refusing a content conflict."""
        held = table.get(rsc.name)
        if held is None:
            table[rsc.name] = rsc
        elif held is not rsc and not EditorMaterialDataTable._check_same_content(
            held, rsc
        ):
            raise EditorMaterialError(
                f"two rscs named '{rsc.name}' carry different content"
            )

    @staticmethod
    def _check_same_content(a: object, b: object) -> bool:
        """Return whether two rscs save to the same bytes."""
        digests = []
        for rsc in (a, b):
            handle, path = tempfile.mkstemp(suffix=".rscb")
            os.close(handle)
            try:
                rsc.save(path)
                with open(path, "rb") as saved:
                    digests.append(hashlib.sha256(saved.read()).hexdigest())
            finally:
                os.remove(path)
        return digests[0] == digests[1]
