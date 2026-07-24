# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Direction vocabulary shared by every create signature.

A direction is either a vector or one of the six axis names, which mean
exactly the axis unit vectors. As a face selector a direction is in the named
node's own local frame; as a free ray or sweep direction it is in the frame
of the draft whose method takes it.
"""

from typing import Literal, Sequence, Union

import numpy as np

from rise_origin import RVec3rf

__all__ = ["DirName", "Dir"]

DirName = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
Dir = Union[RVec3rf, Sequence[float], np.ndarray, DirName]
