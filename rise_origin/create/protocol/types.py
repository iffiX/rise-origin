# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The settings vocabulary of the module protocol.

A module declares typed setting fields with ranges and mandatory defaults;
the schema validates values, fills defaults, and refuses strangers. Every
schema carries an automatic ``seed`` setting, so the seed rides the identity
through the canonical settings json and a module builds its own random
stream from ``settings.seed``.
"""

import json
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ..constants import VOXEL_SIZE_SCALE

__all__ = [
    "SettingsError",
    "Float",
    "Int",
    "Bool",
    "Str",
    "Choice",
    "FloatList",
    "Bounds",
    "VoxelSize",
    "Settings",
    "SettingsSchema",
    "settings",
]

_SEED_MAX = 2**31 - 1


class SettingsError(RuntimeError):
    """A settings declaration or value that breaks the schema."""


class Float:
    """A float setting with an inclusive range and a default.

    Args:
        min: The smallest allowed value, inclusive.
        max: The largest allowed value, inclusive.
        default: The value used when the setting is left unset.
    """

    def __init__(self, min: float, max: float, *, default: float) -> None:
        self.min = float(min)
        self.max = float(max)
        self.default = float(default)

    def validate(self, name: str, value: object) -> float:
        """Return the checked value.

        Args:
            name: The setting name, used in the error message.
            value: The raw value to check and convert.
        """
        number = float(value)
        if not self.min <= number <= self.max:
            raise SettingsError(
                f"setting '{name}'={number} outside [{self.min}, {self.max}]"
            )
        return number

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {
            "type": "float",
            "min": self.min,
            "max": self.max,
            "default": self.default,
        }


class Int:
    """An integer setting with an inclusive range and a default.

    Args:
        min: The smallest allowed value, inclusive.
        max: The largest allowed value, inclusive.
        default: The value used when the setting is left unset.
    """

    def __init__(self, min: int, max: int, *, default: int) -> None:
        self.min = int(min)
        self.max = int(max)
        self.default = int(default)

    def validate(self, name: str, value: object) -> int:
        """Return the checked value.

        Args:
            name: The setting name, used in the error message.
            value: The raw value to check and convert.
        """
        number = int(value)
        if not self.min <= number <= self.max:
            raise SettingsError(
                f"setting '{name}'={number} outside [{self.min}, {self.max}]"
            )
        return number

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {
            "type": "int",
            "min": self.min,
            "max": self.max,
            "default": self.default,
        }


class Bool:
    """A boolean setting with a default.

    Args:
        default: The value used when the setting is left unset.
    """

    def __init__(self, *, default: bool) -> None:
        self.default = bool(default)

    def validate(self, name: str, value: object) -> bool:
        """Return the checked value.

        Args:
            name: The setting name, used in the error message.
            value: The raw value to check and convert.
        """
        if not isinstance(value, (bool, int)):
            raise SettingsError(f"setting '{name}' must be a bool")
        return bool(value)

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {"type": "bool", "default": self.default}


class Str:
    """A free string setting with a default.

    Args:
        default: The value used when the setting is left unset.
    """

    def __init__(self, *, default: str) -> None:
        self.default = str(default)

    def validate(self, name: str, value: object) -> str:
        """Return the checked value.

        Args:
            name: The setting name, used in the error message.
            value: The raw value to check and convert.
        """
        return str(value)

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {"type": "str", "default": self.default}


class Choice:
    """A setting picking one of a fixed set of options.

    Args:
        options: The allowed values.
        default: The value used when the setting is left unset; it must be
            one of the options.
    """

    def __init__(self, *options: str, default: str) -> None:
        if default not in options:
            raise SettingsError(f"default '{default}' not among the options")
        self.options = tuple(options)
        self.default = default

    def validate(self, name: str, value: object) -> str:
        """Return the checked value.

        Args:
            name: The setting name, used in the error message.
            value: The raw value to check and convert.
        """
        if value not in self.options:
            raise SettingsError(
                f"setting '{name}'='{value}' not among {list(self.options)}"
            )
        return str(value)

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {
            "type": "choice",
            "options": list(self.options),
            "default": self.default,
        }


class FloatList:
    """A list-of-floats setting with a per-item range and a default.

    Multi-body modules use this for ``voxel_size``, one value per structure
    body. ``length`` pins the list to an exact size when given.

    Args:
        min: The smallest allowed value for each item, inclusive.
        max: The largest allowed value for each item, inclusive.
        default: The value used when the setting is left unset.
        length: The exact item count required, or None for any length.
    """

    def __init__(
        self,
        min: float,
        max: float,
        *,
        default: Sequence[float],
        length: Optional[int] = None,
    ) -> None:
        self.min = float(min)
        self.max = float(max)
        self.length = None if length is None else int(length)
        self.default = self.validate("default", default)

    def validate(self, name: str, value: object) -> Tuple[float, ...]:
        """Return the checked values as a tuple.

        Args:
            name: The setting name, used in the error message.
            value: The raw sequence of values to check and convert.
        """
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise SettingsError(f"setting '{name}' must be a sequence of floats")
        values = tuple(float(item) for item in value)
        if self.length is not None and len(values) != self.length:
            raise SettingsError(
                f"setting '{name}' needs exactly {self.length} values, "
                f"got {len(values)}"
            )
        for item in values:
            if not self.min <= item <= self.max:
                raise SettingsError(
                    f"setting '{name}' value {item} outside [{self.min}, {self.max}]"
                )
        return values

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {
            "type": "float_list",
            "min": self.min,
            "max": self.max,
            "length": self.length,
            "default": list(self.default),
        }


class Bounds:
    """An axis-aligned 3D bounds setting, a min and a max corner.

    Mirrors the engine's ``RBound3rf``: a value is the min corner and the
    max corner in meters. Accepts an ``RBound3rf``, a flat sequence of
    six numbers (min x, min y, min z, max x, max y, max z), or a pair of
    three-number corners. The validated value is a tuple of the two
    corner tuples.

    Args:
        min: The smallest allowed coordinate on any axis, inclusive.
        max: The largest allowed coordinate on any axis, inclusive.
        default: The value used when the setting is left unset.
    """

    def __init__(self, min: float, max: float, *, default: object) -> None:
        self.min = float(min)
        self.max = float(max)
        self.default = self.validate("default", default)

    def validate(
        self, name: str, value: object
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return the checked min and max corners.

        Args:
            name: The setting name, used in the error message.
            value: The raw bounds to check and convert.
        """
        low, high = Bounds._read_corners(name, value)
        for corner, tag in ((low, "min"), (high, "max")):
            for item in corner:
                if not self.min <= item <= self.max:
                    raise SettingsError(
                        f"setting '{name}' {tag} corner value {item} outside "
                        f"[{self.min}, {self.max}]"
                    )
        for axis in range(3):
            if low[axis] > high[axis]:
                raise SettingsError(
                    f"setting '{name}' min corner above max corner on axis {axis}"
                )
        return (low, high)

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {
            "type": "bounds",
            "min": self.min,
            "max": self.max,
            "default": [list(self.default[0]), list(self.default[1])],
        }

    @staticmethod
    def _read_corners(name: str, value: object):
        """Return the min and max corner tuples read from a raw value.

        Args:
            name: The setting name, used in the error message.
            value: An RBound3rf, six numbers, or two three-number corners.
        """
        if hasattr(value, "min_x") and hasattr(value, "max_z"):
            return (
                (float(value.min_x), float(value.min_y), float(value.min_z)),
                (float(value.max_x), float(value.max_y), float(value.max_z)),
            )
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise SettingsError(
                f"setting '{name}' must be a bounds, six numbers, or two corners"
            )
        items = list(value)
        if len(items) == 6 and all(isinstance(v, (int, float)) for v in items):
            return (
                tuple(float(v) for v in items[0:3]),
                tuple(float(v) for v in items[3:6]),
            )
        if len(items) == 2:
            low, high = items
            return (
                tuple(float(v) for v in low),
                tuple(float(v) for v in high),
            )
        raise SettingsError(
            f"setting '{name}' needs six numbers or two three-number corners"
        )


class VoxelSize:
    """A voxel size choice from the create layer's one size ladder.

    Callers pass a unit text option such as ``"2cm"``; the ladder is
    ``constants.VOXEL_SIZE_SCALE``, 1-2-5 steps from ``"1mm"`` to
    ``"10m"``. Validation refuses anything off the ladder or outside this
    field's range and returns the size in meters, so module bodies do
    math directly and requires declare the exact same floats.

    Args:
        default: The unit text used when the setting is left unset.
        min: The finest allowed option; None allows the ladder's finest.
        max: The coarsest allowed option; None allows the ladder's
            coarsest.
    """

    def __init__(
        self,
        *,
        default: str,
        min: Optional[str] = None,
        max: Optional[str] = None,
    ) -> None:
        ladder = list(VOXEL_SIZE_SCALE)
        for text in (default, min, max):
            if text is not None and text not in VOXEL_SIZE_SCALE:
                raise SettingsError(
                    f"'{text}' is not on the voxel size ladder "
                    f"({', '.join(ladder)})"
                )
        first = 0 if min is None else ladder.index(min)
        last = len(ladder) - 1 if max is None else ladder.index(max)
        if first > last:
            raise SettingsError(f"voxel size range '{min}'..'{max}' is empty")
        self.options: Tuple[str, ...] = tuple(ladder[first : last + 1])
        self.default_text = default
        self.default = self.validate("default", default)

    def validate(self, name: str, value: object) -> float:
        """Return the chosen size in meters.

        Args:
            name: The setting name, used in the error message.
            value: The unit text to check, such as ``"2cm"``.
        """
        if not isinstance(value, str) or value not in self.options:
            raise SettingsError(
                f"setting '{name}'={value!r} must be one of " + ", ".join(self.options)
            )
        return VOXEL_SIZE_SCALE[value]

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {
            "type": "voxel_size",
            "options": list(self.options),
            "default": self.default_text,
        }


class Settings(Mapping):
    """Validated setting values with attribute access."""

    def __init__(self, values: Dict[str, object]) -> None:
        self._values = dict(values)

    def __getattr__(self, name: str) -> object:
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __getitem__(self, name: str) -> object:
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_json(self) -> str:
        """Return the canonical json of the values, sorted keys."""
        return json.dumps(self._values, sort_keys=True)


class SettingsSchema:
    """The declared settings of a module: names, types, ranges, defaults.

    Every schema automatically carries a ``seed`` integer setting with a
    default of zero; declaring an own ``seed`` field is refused.
    """

    def __init__(self, **fields: object) -> None:
        if "seed" in fields:
            raise SettingsError("'seed' is added automatically; do not declare it")
        for name, kind in fields.items():
            if not hasattr(kind, "validate") or not hasattr(kind, "default"):
                raise SettingsError(
                    f"setting '{name}' needs a typed field with a default"
                )
        self.fields = dict(fields)
        self.fields["seed"] = Int(0, _SEED_MAX, default=0)

    def validate(self, values: Optional[Mapping]) -> Settings:
        """Return validated values with defaults filled, refusing strangers.

        Args:
            values: The raw setting values to check, or None for all
                defaults.
        """
        given = dict(values or {})
        unknown = set(given) - set(self.fields)
        if unknown:
            raise SettingsError(f"unknown settings {sorted(unknown)}")
        checked = {}
        for name, kind in self.fields.items():
            if name in given:
                checked[name] = kind.validate(name, given[name])
            else:
                checked[name] = kind.default
        return Settings(checked)

    def to_json(self) -> dict:
        """Return the json description of every field."""
        return {name: kind.to_json() for name, kind in self.fields.items()}


settings = SettingsSchema
