# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The identity of a module-born Draft: the realization cache key.

Purity makes the identity a content address: the same identity is the same
bytes, every time, everywhere. The seed lives inside the settings, so the
identity is the module at a version plus the canonical settings json.
Name and version parsing go through :class:`ModuleReference`.
"""

from dataclasses import dataclass

from .module_reference import ModuleReference

__all__ = ["Identity"]


@dataclass(frozen=True)
class Identity:
    """The replayable identity of a module-born Draft.

    Attributes:
        module: Full module name such as ``rise_origin.nature.tree.maple``.
        version: The concrete module version at build time.
        settings_json: Canonical json of the validated settings, sorted keys,
            seed included.
    """

    module: str
    version: str
    settings_json: str

    def get_reference(self) -> ModuleReference:
        """Return the module reference of this identity, version included."""
        reference = ModuleReference.from_text(self.module)
        return ModuleReference(
            reference.organization,
            reference.area,
            reference.group,
            reference.kind,
            version_text=self.version,
        )

    def make_hash_text(self) -> str:
        """Return the text the cache hashes into the content address."""
        return f"{self.module}@{self.version}|{self.settings_json}"

    def to_json(self) -> dict:
        """Return the json form of this identity."""
        return {
            "module": self.module,
            "version": self.version,
            "settings": self.settings_json,
        }

    @staticmethod
    def from_json(identity_json: dict) -> "Identity":
        """Return the identity of its json form.

        Args:
            identity_json: The json form of an identity, as :meth:`to_json`
                returns it.
        """
        return Identity(
            identity_json["module"],
            identity_json["version"],
            identity_json["settings"],
        )
