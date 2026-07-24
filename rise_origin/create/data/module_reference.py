# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The module reference: one parsed name with the version text behind it.

Every module name has four levels, ``organization.area.group.kind``, such as
``rise_origin.nature.tree.maple``; a three-level name defaults to the
``rise_origin`` organization. The text behind ``@`` is the version part: a
concrete version on an identity, a version specifier on a require, or ``*``
for no constraint at all.
"""

from dataclasses import dataclass
from typing import Optional

from rise_origin import RS_DataException, RS_Version

from ..constants import DEFAULT_ORGANIZATION

__all__ = ["ModuleReference"]


@dataclass(frozen=True)
class ModuleReference:
    """One parsed module reference.

    Attributes:
        organization: The publishing organization, such as ``rise_origin``.
        area: The subject area, such as ``nature``.
        group: The group inside the area, such as ``tree``.
        kind: The real-world kind, such as ``maple``.
        version_text: The raw text behind ``@``: a concrete version, a
            version specifier, or None when the reference names no version.
    """

    organization: str
    area: str
    group: str
    kind: str
    version_text: Optional[str] = None

    def get_name(self) -> str:
        """Return the full four-level name without the version part."""
        return f"{self.organization}.{self.area}.{self.group}.{self.kind}"

    def to_text(self) -> str:
        """Return the full reference text, version part included."""
        name = self.get_name()
        if self.version_text is None:
            return name
        return f"{name}@{self.version_text}"

    def is_version_matching(self, version: RS_Version) -> bool:
        """Return whether a version satisfies the version part as a specifier.

        No version part and ``*`` both mean no constraint, so every version
        satisfies them; anything else answers through the engine's own
        specifier grammar.

        Args:
            version: The concrete version to test against the version part.
        """
        if self.version_text is None or self.version_text == "*":
            return True
        return version.is_matching(self.version_text)

    @staticmethod
    def from_text(text: str) -> "ModuleReference":
        """Return the parsed reference of a name text.

        A three-level name defaults the organization to ``rise_origin``;
        anything but three or four non-empty dotted levels is refused.

        Args:
            text: The dotted module name to parse, with an optional ``@``
                version part.
        """
        base, _, version_text = text.partition("@")
        parts = base.split(".")
        if len(parts) == 3:
            parts = [DEFAULT_ORGANIZATION] + parts
        if len(parts) != 4 or not all(parts):
            raise ValueError(
                f"'{text}' is not an organization.area.group.kind module name"
            )
        return ModuleReference(*parts, version_text=(version_text or None))

    @staticmethod
    def parse_version(text: str) -> RS_Version:
        """Return the parsed engine version of a concrete version string.

        Args:
            text: The concrete version string to parse; a version
                specifier such as ``^1`` is refused, since the version
                part of a reference may hold either.

        Raises:
            ValueError: The text is not a concrete version.
        """
        version = RS_Version()
        try:
            version.from_string(text)
        except RS_DataException as error:
            raise ValueError(f"'{text}' is not a concrete version") from error
        return version
