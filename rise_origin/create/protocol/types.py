# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The settings vocabulary of the module protocol.

A module declares typed setting fields with ranges and mandatory defaults;
the schema validates values, fills defaults, and refuses strangers. Every
schema carries an automatic ``seed`` setting, so the seed rides the identity
through the canonical settings json and a module builds its own random
stream from ``settings.seed``.
"""

import base64
import json
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..constants import VOXEL_SIZE_SCALE

__all__ = [
    "SettingsError",
    "Float",
    "Int",
    "Bool",
    "Str",
    "Choice",
    "Vector",
    "Bounds",
    "Array1D",
    "Array2D",
    "Array3D",
    "StaticArray1D",
    "StaticArray2D",
    "StaticArray3D",
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
        doc: The per-setting help text, shown in the module manual.
    """

    def __init__(
        self, min: float, max: float, *, default: float, doc: str = ""
    ) -> None:
        self.min = float(min)
        self.max = float(max)
        self.default = float(default)
        self.doc = str(doc)

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
            "doc": self.doc,
        }


class Int:
    """An integer setting with an inclusive range and a default.

    Args:
        min: The smallest allowed value, inclusive.
        max: The largest allowed value, inclusive.
        default: The value used when the setting is left unset.
        doc: The per-setting help text, shown in the module manual.
    """

    def __init__(self, min: int, max: int, *, default: int, doc: str = "") -> None:
        self.min = int(min)
        self.max = int(max)
        self.default = int(default)
        self.doc = str(doc)

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
            "doc": self.doc,
        }


class Bool:
    """A boolean setting with a default.

    Args:
        default: The value used when the setting is left unset.
        doc: The per-setting help text, shown in the module manual.
    """

    def __init__(self, *, default: bool, doc: str = "") -> None:
        self.default = bool(default)
        self.doc = str(doc)

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
        return {"type": "bool", "default": self.default, "doc": self.doc}


class Str:
    """A free string setting with a default.

    Args:
        default: The value used when the setting is left unset.
        doc: The per-setting help text, shown in the module manual.
    """

    def __init__(self, *, default: str, doc: str = "") -> None:
        self.default = str(default)
        self.doc = str(doc)

    def validate(self, name: str, value: object) -> str:
        """Return the checked value.

        Args:
            name: The setting name, used in the error message.
            value: The raw value to check and convert.
        """
        return str(value)

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {"type": "str", "default": self.default, "doc": self.doc}


class Choice:
    """A setting picking one of a fixed set of options.

    Args:
        options: The allowed values.
        default: The value used when the setting is left unset; it must be
            one of the options.
        doc: The per-setting help text, shown in the module manual.
    """

    def __init__(self, *options: str, default: str, doc: str = "") -> None:
        if default not in options:
            raise SettingsError(f"default '{default}' not among the options")
        self.options = tuple(options)
        self.default = default
        self.doc = str(doc)

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
            "doc": self.doc,
        }


class Vector:
    """A 3D vector setting, three numbers on x, y, z.

    Mirrors the engine's ``RVec3rf``. Accepts an ``RVec3rf`` or a
    three-number sequence; the validated value is a tuple of three
    floats. A direction the reader normalizes itself.

    Args:
        min: The smallest allowed coordinate on any axis, inclusive.
        max: The largest allowed coordinate on any axis, inclusive.
        default: The value used when the setting is left unset.
        doc: The per-setting help text, shown in the module manual.
    """

    def __init__(
        self, min: float, max: float, *, default: object, doc: str = ""
    ) -> None:
        self.min = float(min)
        self.max = float(max)
        self.doc = str(doc)
        self.default = self.validate("default", default)

    def validate(self, name: str, value: object) -> Tuple[float, float, float]:
        """Return the checked three-number vector as a tuple.

        Args:
            name: The setting name, used in the error message.
            value: An RVec3rf or a three-number sequence.
        """
        parts = Vector._read(name, value)
        for item in parts:
            if not self.min <= item <= self.max:
                raise SettingsError(
                    f"setting '{name}' value {item} outside [{self.min}, {self.max}]"
                )
        return parts

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {
            "type": "vector",
            "min": self.min,
            "max": self.max,
            "default": list(self.default),
            "doc": self.doc,
        }

    @staticmethod
    def _read(name: str, value: object) -> Tuple[float, float, float]:
        """Return the three components read from a raw value.

        Args:
            name: The setting name, used in the error message.
            value: An RVec3rf or a three-number sequence.
        """
        if hasattr(value, "x") and hasattr(value, "z"):
            return (float(value.x), float(value.y), float(value.z))
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise SettingsError(f"setting '{name}' must be a vector or three numbers")
        items = list(value)
        if len(items) != 3:
            raise SettingsError(f"setting '{name}' needs three numbers")
        return tuple(float(item) for item in items)


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
        doc: The per-setting help text, shown in the module manual.
    """

    def __init__(
        self, min: float, max: float, *, default: object, doc: str = ""
    ) -> None:
        self.min = float(min)
        self.max = float(max)
        self.doc = str(doc)
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
            "doc": self.doc,
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


class _Array:
    """The shared cell checks and byte coding of the array settings.

    A variable field bounds its total cells with ``cell_cap``; a static
    field pins the exact ``shape`` instead. Exactly one of the two is
    set by the public subclasses.
    """

    RANK = 0
    KIND_NAME = ""
    DTYPE_BYTE_CODES = {"uint8": "<u1", "int32": "<i4", "float32": "<f4"}

    def __init__(
        self,
        *,
        dtype: str,
        min: Optional[float],
        max: Optional[float],
        cell_cap: Optional[int],
        shape: Optional[Sequence[int]],
        default: object,
        doc: str,
    ) -> None:
        if dtype != "bool" and dtype not in self.DTYPE_BYTE_CODES:
            raise SettingsError(
                f"dtype '{dtype}' is not one of bool, "
                + ", ".join(self.DTYPE_BYTE_CODES)
            )
        if dtype == "bool" and (min is not None or max is not None):
            raise SettingsError("bool cells take no min or max")
        self.dtype = dtype
        self.min = None if min is None else float(min)
        self.max = None if max is None else float(max)
        self.cell_cap = None if cell_cap is None else int(cell_cap)
        self.shape = None if shape is None else self._make_checked_shape(shape)
        self.doc = str(doc)
        self.is_none_allowed = default is None
        self.default = None if default is None else self.validate("default", default)

    def validate(self, name: str, value: object) -> Optional[dict]:
        """Return the cells in their canonical json form.

        Args:
            name: The setting name, used in the error message.
            value: A numpy array, nested lists, the canonical json dict,
                or None when the default is None.
        """
        if value is None:
            if not self.is_none_allowed:
                raise SettingsError(f"setting '{name}' must be set")
            return None
        if isinstance(value, dict):
            cells = self._make_cells_from_json(name, value)
        else:
            cells = self._make_cells_from_data(name, value)
        self._check_cell_values(name, cells)
        return {
            "cell_num": [int(side) for side in cells.shape],
            "dtype": self.dtype,
            "data": self._encode_cells(cells),
        }

    def make_value(self, stored: Optional[dict]) -> object:
        """Return the stored cells as a numpy array of the declared dtype.

        Args:
            stored: The canonical json form validate returned, or None.
        """
        if stored is None:
            return None
        shape = tuple(stored["cell_num"])
        payload = base64.b64decode(stored["data"])
        if self.dtype == "bool":
            flat = np.unpackbits(
                np.frombuffer(payload, np.uint8), count=math.prod(shape)
            )
            return flat.astype(np.bool_).reshape(shape)
        code = self.DTYPE_BYTE_CODES[self.dtype]
        return np.frombuffer(payload, dtype=code).reshape(shape).copy()

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {
            "type": self.KIND_NAME,
            "dtype": self.dtype,
            "min": self.min,
            "max": self.max,
            "cell_cap": self.cell_cap,
            "shape": None if self.shape is None else list(self.shape),
            "default": self.default,
            "doc": self.doc,
        }

    def _make_checked_shape(self, shape: Sequence[int]) -> Tuple[int, ...]:
        """Return the declared static shape checked against the rank."""
        sides = tuple(int(side) for side in shape)
        if len(sides) != self.RANK:
            raise SettingsError(f"shape {sides} needs {self.RANK} dimensions")
        if any(side < 1 for side in sides):
            raise SettingsError(f"shape {sides} has an empty dimension")
        return sides

    def _make_cells_from_data(self, name: str, value: object) -> "np.ndarray":
        """Return checked cells read from an array or nested lists."""
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise SettingsError(f"setting '{name}' is not an array of cells") from error
        self._check_cell_shape(name, raw.shape)
        if self.dtype == "bool":
            if raw.dtype == np.bool_:
                return raw
            if raw.dtype.kind in "iu":
                if not np.isin(raw, (0, 1)).all():
                    raise SettingsError(f"setting '{name}' bool cells must be 0 or 1")
                return raw.astype(np.bool_)
            raise SettingsError(
                f"setting '{name}' bool cells need bool or 0/1 integer values"
            )
        code = self.DTYPE_BYTE_CODES[self.dtype]
        if self.dtype == "float32":
            if raw.dtype.kind not in "iuf":
                raise SettingsError(f"setting '{name}' float32 cells need numbers")
            return raw.astype(code)
        if raw.dtype.kind not in "iub":
            raise SettingsError(f"setting '{name}' {self.dtype} cells need integers")
        limits = np.iinfo(code)
        if raw.size and (raw.min() < limits.min or raw.max() > limits.max):
            raise SettingsError(
                f"setting '{name}' cells outside the {self.dtype} range"
            )
        return raw.astype(code)

    def _make_cells_from_json(self, name: str, value: dict) -> "np.ndarray":
        """Return checked cells read back from the canonical json form."""
        keys = {"cell_num", "dtype", "data"}
        if set(value) != keys:
            raise SettingsError(f"setting '{name}' json needs keys {sorted(keys)}")
        if value["dtype"] != self.dtype:
            raise SettingsError(
                f"setting '{name}' dtype '{value['dtype']}' is not '{self.dtype}'"
            )
        sides = value["cell_num"]
        if not isinstance(sides, (list, tuple)) or not all(
            isinstance(side, int) and not isinstance(side, bool) for side in sides
        ):
            raise SettingsError(f"setting '{name}' cell_num needs integer sides")
        shape = tuple(sides)
        self._check_cell_shape(name, shape)
        try:
            payload = base64.b64decode(value["data"], validate=True)
        except (TypeError, ValueError) as error:
            raise SettingsError(f"setting '{name}' data is not base64") from error
        cell_num = math.prod(shape)
        if self.dtype == "bool":
            if len(payload) != (cell_num + 7) // 8:
                raise SettingsError(f"setting '{name}' data length is wrong")
            flat = np.unpackbits(np.frombuffer(payload, np.uint8), count=cell_num)
            return flat.astype(np.bool_).reshape(shape)
        code = self.DTYPE_BYTE_CODES[self.dtype]
        if len(payload) != cell_num * np.dtype(code).itemsize:
            raise SettingsError(f"setting '{name}' data length is wrong")
        return np.frombuffer(payload, dtype=code).reshape(shape)

    def _check_cell_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        """Check a value's shape against the declared shape or cap."""
        if self.shape is not None:
            if tuple(shape) != self.shape:
                raise SettingsError(
                    f"setting '{name}' needs shape {self.shape}, got {tuple(shape)}"
                )
            return
        if len(shape) != self.RANK:
            raise SettingsError(
                f"setting '{name}' needs {self.RANK} dimensions, got {len(shape)}"
            )
        if any(side < 1 for side in shape):
            raise SettingsError(f"setting '{name}' has an empty dimension")
        cell_num = math.prod(shape)
        if cell_num > self.cell_cap:
            raise SettingsError(
                f"setting '{name}' holds {cell_num} cells over the cap {self.cell_cap}"
            )

    def _check_cell_values(self, name: str, cells: "np.ndarray") -> None:
        """Check finiteness and the declared range of the cell values."""
        if self.dtype == "bool":
            return
        if self.dtype == "float32" and not np.isfinite(cells).all():
            raise SettingsError(f"setting '{name}' cells must be finite")
        if self.min is not None and cells.size and float(cells.min()) < self.min:
            raise SettingsError(f"setting '{name}' has cells below {self.min}")
        if self.max is not None and cells.size and float(cells.max()) > self.max:
            raise SettingsError(f"setting '{name}' has cells above {self.max}")

    def _encode_cells(self, cells: "np.ndarray") -> str:
        """Return the deterministic base64 text of the cells."""
        if self.dtype == "bool":
            payload = np.packbits(cells.astype(np.uint8).ravel(order="C")).tobytes()
        else:
            payload = np.ascontiguousarray(cells).tobytes()
        return base64.b64encode(payload).decode("ascii")


class Array1D(_Array):
    """A variable-size 1D array setting of cells with a declared dtype.

    The cells are dimensionless data; the consuming module gives them
    meaning. The canonical json form rides the identity, journal, and
    cache text, so ``cell_cap`` bounds that cost.

    Args:
        dtype: The element dtype, one of ``"bool"``, ``"uint8"``,
            ``"int32"``, ``"float32"``.
        min: The smallest allowed cell value, or None for no floor;
            refused for bool cells.
        max: The largest allowed cell value, or None for no ceiling;
            refused for bool cells.
        cell_cap: The largest allowed total cell count.
        default: The value used when the setting is left unset; None
            makes the setting optional and stays None.
        doc: The per-setting help text, shown in the module manual.
    """

    RANK = 1
    KIND_NAME = "array1d"

    def __init__(
        self,
        *,
        dtype: str = "float32",
        min: Optional[float] = None,
        max: Optional[float] = None,
        cell_cap: int = 4_096,
        default: object = None,
        doc: str = "",
    ) -> None:
        super().__init__(
            dtype=dtype,
            min=min,
            max=max,
            cell_cap=cell_cap,
            shape=None,
            default=default,
            doc=doc,
        )


class Array2D(_Array):
    """A variable-size 2D grid setting of cells with a declared dtype.

    The grid is dimensionless: nothing in it carries meters, and the
    consuming module stretches it over its own size settings, each
    dimension divided independently. The canonical json form rides the
    identity, journal, and cache text, so ``cell_cap`` bounds that cost.

    Args:
        dtype: The element dtype, one of ``"bool"``, ``"uint8"``,
            ``"int32"``, ``"float32"``.
        min: The smallest allowed cell value, or None for no floor;
            refused for bool cells.
        max: The largest allowed cell value, or None for no ceiling;
            refused for bool cells.
        cell_cap: The largest allowed total cell count.
        default: The value used when the setting is left unset; None
            makes the setting optional and stays None.
        doc: The per-setting help text, shown in the module manual.
    """

    RANK = 2
    KIND_NAME = "array2d"

    def __init__(
        self,
        *,
        dtype: str = "float32",
        min: Optional[float] = None,
        max: Optional[float] = None,
        cell_cap: int = 65_536,
        default: object = None,
        doc: str = "",
    ) -> None:
        super().__init__(
            dtype=dtype,
            min=min,
            max=max,
            cell_cap=cell_cap,
            shape=None,
            default=default,
            doc=doc,
        )


class Array3D(_Array):
    """A variable-size 3D grid setting of cells with a declared dtype.

    The grid is dimensionless: nothing in it carries meters, and the
    consuming module stretches it over its own size settings, each
    dimension divided independently; a bool grid over a module's box is
    the occupancy form. The canonical json form rides the identity,
    journal, and cache text, so ``cell_cap`` bounds that cost.

    Args:
        dtype: The element dtype, one of ``"bool"``, ``"uint8"``,
            ``"int32"``, ``"float32"``.
        min: The smallest allowed cell value, or None for no floor;
            refused for bool cells.
        max: The largest allowed cell value, or None for no ceiling;
            refused for bool cells.
        cell_cap: The largest allowed total cell count.
        default: The value used when the setting is left unset; None
            makes the setting optional and stays None.
        doc: The per-setting help text, shown in the module manual.
    """

    RANK = 3
    KIND_NAME = "array3d"

    def __init__(
        self,
        *,
        dtype: str = "bool",
        min: Optional[float] = None,
        max: Optional[float] = None,
        cell_cap: int = 262_144,
        default: object = None,
        doc: str = "",
    ) -> None:
        super().__init__(
            dtype=dtype,
            min=min,
            max=max,
            cell_cap=cell_cap,
            shape=None,
            default=default,
            doc=doc,
        )


class StaticArray1D(_Array):
    """A fixed-shape 1D array setting of cells with a declared dtype.

    The declared ``shape`` is the exact shape every value must have;
    anything else is refused. The cells are dimensionless data; the
    consuming module gives them meaning.

    Args:
        shape: The exact required shape, one side.
        dtype: The element dtype, one of ``"bool"``, ``"uint8"``,
            ``"int32"``, ``"float32"``.
        min: The smallest allowed cell value, or None for no floor;
            refused for bool cells.
        max: The largest allowed cell value, or None for no ceiling;
            refused for bool cells.
        default: The value used when the setting is left unset; None
            makes the setting optional and stays None.
        doc: The per-setting help text, shown in the module manual.
    """

    RANK = 1
    KIND_NAME = "static_array1d"

    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: str = "float32",
        min: Optional[float] = None,
        max: Optional[float] = None,
        default: object = None,
        doc: str = "",
    ) -> None:
        super().__init__(
            dtype=dtype,
            min=min,
            max=max,
            cell_cap=None,
            shape=shape,
            default=default,
            doc=doc,
        )


class StaticArray2D(_Array):
    """A fixed-shape 2D grid setting of cells with a declared dtype.

    The declared ``shape`` is the exact shape every value must have;
    anything else is refused. The grid is dimensionless; the consuming
    module stretches it over its own size settings.

    Args:
        shape: The exact required shape, two sides.
        dtype: The element dtype, one of ``"bool"``, ``"uint8"``,
            ``"int32"``, ``"float32"``.
        min: The smallest allowed cell value, or None for no floor;
            refused for bool cells.
        max: The largest allowed cell value, or None for no ceiling;
            refused for bool cells.
        default: The value used when the setting is left unset; None
            makes the setting optional and stays None.
        doc: The per-setting help text, shown in the module manual.
    """

    RANK = 2
    KIND_NAME = "static_array2d"

    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: str = "float32",
        min: Optional[float] = None,
        max: Optional[float] = None,
        default: object = None,
        doc: str = "",
    ) -> None:
        super().__init__(
            dtype=dtype,
            min=min,
            max=max,
            cell_cap=None,
            shape=shape,
            default=default,
            doc=doc,
        )


class StaticArray3D(_Array):
    """A fixed-shape 3D grid setting of cells with a declared dtype.

    The declared ``shape`` is the exact shape every value must have;
    anything else is refused. The grid is dimensionless; the consuming
    module stretches it over its own size settings.

    Args:
        shape: The exact required shape, three sides.
        dtype: The element dtype, one of ``"bool"``, ``"uint8"``,
            ``"int32"``, ``"float32"``.
        min: The smallest allowed cell value, or None for no floor;
            refused for bool cells.
        max: The largest allowed cell value, or None for no ceiling;
            refused for bool cells.
        default: The value used when the setting is left unset; None
            makes the setting optional and stays None.
        doc: The per-setting help text, shown in the module manual.
    """

    RANK = 3
    KIND_NAME = "static_array3d"

    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: str = "bool",
        min: Optional[float] = None,
        max: Optional[float] = None,
        default: object = None,
        doc: str = "",
    ) -> None:
        super().__init__(
            dtype=dtype,
            min=min,
            max=max,
            cell_cap=None,
            shape=shape,
            default=default,
            doc=doc,
        )


class VoxelSize:
    """A voxel size choice from the create layer's one size ladder.

    Callers pass a unit text option such as ``"2cm"``; the ladder is
    ``constants.VOXEL_SIZE_SCALE``, 1-2-5 steps from ``"1mm"`` to
    ``"10m"``. Validation refuses anything off the ladder or outside
    this field's range and keeps the unit text as the canonical form;
    module bodies read the size in meters, so requires declare the
    exact same floats.

    Args:
        default: The unit text used when the setting is left unset.
        min: The finest allowed option; None allows the ladder's finest.
        max: The coarsest allowed option; None allows the ladder's
            coarsest.
        doc: The per-setting help text, shown in the module manual.
    """

    def __init__(
        self,
        *,
        default: str,
        min: Optional[str] = None,
        max: Optional[str] = None,
        doc: str = "",
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
        self.doc = str(doc)
        self.default = self.validate("default", default)

    def validate(self, name: str, value: object) -> str:
        """Return the checked unit text.

        Args:
            name: The setting name, used in the error message.
            value: The unit text to check, such as ``"2cm"``.
        """
        if not isinstance(value, str) or value not in self.options:
            raise SettingsError(
                f"setting '{name}'={value!r} must be one of " + ", ".join(self.options)
            )
        return value

    def make_value(self, stored: str) -> float:
        """Return the stored unit text as the size in meters.

        Args:
            stored: The canonical unit text validate returned.
        """
        return VOXEL_SIZE_SCALE[stored]

    def to_json(self) -> dict:
        """Return the json description of this field."""
        return {
            "type": "voxel_size",
            "options": list(self.options),
            "default": self.default,
            "doc": self.doc,
        }


class Settings(Mapping):
    """Validated setting values with attribute access.

    Values live in their canonical json form; a field type with a
    ``make_value`` hook decodes to the value module bodies read on
    first access, and the decoded value is cached per name.
    """

    def __init__(
        self,
        values: Dict[str, object],
        fields: Optional[Dict[str, object]] = None,
    ) -> None:
        self._values = dict(values)
        self._fields = dict(fields or {})
        self._decoded: Dict[str, object] = {}

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __getitem__(self, name: str) -> object:
        if name in self._decoded:
            return self._decoded[name]
        value = self._values[name]
        kind = self._fields.get(name)
        if kind is None or not hasattr(kind, "make_value"):
            return value
        rich = kind.make_value(value)
        self._decoded[name] = rich
        return rich

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_json(self) -> str:
        """Return the canonical json of the values, sorted keys."""
        return json.dumps(self._values, sort_keys=True)


class SettingsSchema:
    """The declared settings of a module: names, types, ranges, defaults.

    A field type's ``validate`` returns the canonical json form and
    accepts that form back unchanged, so an identity json re-validates
    on a cache rebuild; an optional ``make_value`` hook decodes it into
    the value module bodies read. Every schema automatically carries a
    ``seed`` integer setting; declaring an own ``seed`` is refused.
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
        self.fields["seed"] = Int(
            0, _SEED_MAX, default=0, doc="The random seed of the build."
        )

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
        return Settings(checked, self.fields)

    def to_json(self) -> dict:
        """Return the json description of every field."""
        return {name: kind.to_json() for name, kind in self.fields.items()}


settings = SettingsSchema
