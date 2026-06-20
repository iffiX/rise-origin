# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Engine launching and viewing utilities.

``instance`` launches or attaches to a rise-origin world; ``viewer`` opens an
interactive window over a running instance.
"""

from rise_origin.runtime.instance import Instance, InstanceLocal, InstanceRemote

__all__ = ["Instance", "InstanceLocal", "InstanceRemote"]
