# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The compile currency: world updates and the memo of what a consumer saw.

Compiling never returns a whole world; it returns an ordered instruction list
against a memo. The editor's own memo makes compiles incremental; a fresh
empty memo asks for the full emission; consumers holding their own memos each
receive exactly their missing delta.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Set, Tuple

from .pose import Pose

__all__ = [
    "WorldUpdateInstruction",
    "WorldUpdate",
    "CompileMemoEntity",
    "CompileMemo",
]


@dataclass(frozen=True)
class WorldUpdateInstruction:
    """One ordered instruction of a world update.

    Attributes:
        op: The instruction kind.
        rsc: The rsc object to add; ``add`` only.
        entity_name: The target entity name; ``delete`` and ``move`` only.
        pose: The new root-frame pose; ``move`` only.
    """

    op: Literal["add", "delete", "move"]
    rsc: Optional[object] = None
    entity_name: Optional[str] = None
    pose: Optional[Pose] = None


@dataclass(frozen=True)
class WorldUpdate:
    """An ordered list of world update instructions.

    Attributes:
        instructions: The instructions, applied in order.
    """

    instructions: Tuple[WorldUpdateInstruction, ...] = ()


@dataclass(frozen=True)
class CompileMemoEntity:
    """What the memo remembers about one emitted entity.

    Attributes:
        fingerprint: The content fingerprint of the entity as emitted.
        pose: The root-frame pose of the entity as emitted.
    """

    fingerprint: str
    pose: Pose


@dataclass
class CompileMemo:
    """What one consumer has already seen.

    The editor advances the memo it compiles against; a fresh empty memo
    yields the full world emission.

    Attributes:
        entities: Emitted entity names to their remembered content and pose.
        essence_names: Emitted essence names.
        resource_names: Emitted resource names.
        journal_position: The journal cursor the memo has caught up to.
    """

    entities: Dict[str, CompileMemoEntity] = field(default_factory=dict)
    essence_names: Set[str] = field(default_factory=set)
    resource_names: Set[str] = field(default_factory=set)
    journal_position: int = 0
