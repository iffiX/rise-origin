# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The material require: what a node needs and the voxel sizes it needs it at.

Material definitions are plain rsc essences carried in a draft's rscs; there
is no wrapper layer over them. A material module is a package releasing
several materials: ``rise_origin.material.nature.plant`` may release
``leaf`` and ``bark``, each with its own property essences. A require
declares one such need: the module at a version specifier, the material's
name inside it, and the voxel sizes the content is authored at.

The world references a material by joining the module and the material with
a colon, such as ``rise_origin.material.nature.plant:leaf``; voxels and
property essences carry exactly this string.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

from rise_origin import RS_Version

from ..constants import VOXEL_SIZE_DECIMALS
from .module_reference import ModuleReference

__all__ = ["MaterialRequire"]


@dataclass(frozen=True)
class MaterialRequire:
    """One declared pairing of a material and its authoring voxel sizes.

    Attributes:
        module_reference: The material module releasing the material; text
            such as ``rise_origin.material.nature.plant@^1`` parses at
            construction.
        material_reference: The material's name inside the module, such as
            ``leaf``; a require always names exactly one material, since
            pair and optical-only modules enter a world through plain
            module dependencies instead.
        voxel_size: The voxel edge lengths in meters the content is
            authored at; one float or a sequence at construction, always
            stored as a tuple holding at least one size.
    """

    module_reference: Union[ModuleReference, str]
    material_reference: str
    voxel_size: Union[float, Sequence[float], Tuple[float, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.module_reference, ModuleReference):
            object.__setattr__(
                self,
                "module_reference",
                ModuleReference.from_text(self.module_reference),
            )
        if not self.material_reference:
            raise ValueError(
                f"a require of '{self.module_reference.to_text()}' must name "
                "one material"
            )
        size = self.voxel_size
        if isinstance(size, (int, float)):
            object.__setattr__(self, "voxel_size", (float(size),))
        else:
            object.__setattr__(
                self, "voxel_size", tuple(float(value) for value in size)
            )
        if not self.voxel_size:
            raise ValueError(
                f"a require of '{self.get_world_material_reference()}' must "
                "carry at least one voxel size"
            )

    def get_world_material_reference(self) -> str:
        """Return the reference string the compiled world uses.

        The name joins the module and the material with a colon, such as
        ``rise_origin.material.nature.plant:leaf``; voxels and property
        essences reference materials by exactly this string.
        """
        return f"{self.module_reference.get_name()}:{self.material_reference}"

    def is_version_matching(self, version: RS_Version) -> bool:
        """Return whether a concrete module version satisfies this require.

        Args:
            version: The concrete module version checked against this
                require's version specifier.
        """
        return self.module_reference.is_version_matching(version)

    def is_superset_of(self, other: "MaterialRequire") -> bool:
        """Return whether this require covers another require.

        Covering means both name the same module and material, version
        parts aside, and every one of the other's voxel sizes is among
        this require's sizes, compared rounded.

        Args:
            other: The require checked against this one.
        """
        if other.module_reference.get_name() != self.module_reference.get_name():
            return False
        if other.material_reference != self.material_reference:
            return False
        held = {MaterialRequire.round_voxel_size(size) for size in self.voxel_size}
        return all(
            MaterialRequire.round_voxel_size(size) in held for size in other.voxel_size
        )

    def merge(self, other: "MaterialRequire") -> "MaterialRequire":
        """Return one require holding both requires' voxel sizes.

        Merging is only for requires carrying the same module reference,
        version part included, and the same material, so ``plant@1.0.0``
        and ``plant@0.2.0`` never merge. Sizes union with rounded
        comparison, this require's sizes first.

        Args:
            other: The require merged in; its module reference and
                material must equal this one's.

        Raises:
            ValueError: The module references or materials differ.
        """
        if (
            other.module_reference != self.module_reference
            or other.material_reference != self.material_reference
        ):
            raise ValueError(
                f"cannot merge require '{other.module_reference.to_text()}:"
                f"{other.material_reference}' into "
                f"'{self.module_reference.to_text()}:{self.material_reference}'"
            )
        sizes = list(self.voxel_size)
        held = {MaterialRequire.round_voxel_size(size) for size in sizes}
        for size in other.voxel_size:
            if MaterialRequire.round_voxel_size(size) not in held:
                sizes.append(size)
                held.add(MaterialRequire.round_voxel_size(size))
        return MaterialRequire(
            self.module_reference, self.material_reference, tuple(sizes)
        )

    def to_json(self) -> dict:
        """Return the json form of this require."""
        return {
            "module": self.module_reference.to_text(),
            "material": self.material_reference,
            "voxel_size": list(self.voxel_size),
        }

    @staticmethod
    def from_json(require_json: dict) -> "MaterialRequire":
        """Return the require of its json form.

        Args:
            require_json: The json form of a require, as :meth:`to_json`
                returns it.
        """
        return MaterialRequire(
            require_json["module"],
            require_json["material"],
            require_json["voxel_size"],
        )

    @staticmethod
    def round_voxel_size(size: float) -> float:
        """Return the voxel size rounded for comparison across float32.

        Args:
            size: The voxel edge length in meters to round.
        """
        return round(float(size), VOXEL_SIZE_DECIMALS)
