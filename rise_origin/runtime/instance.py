# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Launchable rise-origin instances a viewer (or your own code) attaches to.

An *instance* is the thing that runs, or connects to, a rise-origin world.
:class:`InstanceLocal` owns a local engine bound to one device and runs a world
in process. :class:`InstanceRemote` (not yet implemented) attaches to a world
running in another process or on another machine. Both expose the same small
surface so a :class:`~rise_origin.runtime.viewer.Viewer` can drive either
transparently.

Most users never construct these directly. A :class:`Viewer` makes one for you.
But they are first-class: you can launch and drive a world with just an
:class:`InstanceLocal`.
"""

from typing import Callable, Optional, Union

import rise_origin as rs

__all__ = ["Instance", "InstanceLocal", "InstanceRemote"]


class Instance:
    """Base class for a launchable or attachable rise-origin world."""


class InstanceLocal(Instance):
    """A local rise-origin engine bound to one compute device.

    Owns a :class:`~rise_origin.RiseOrigin`, runs a world in process via
    :meth:`hello`, and exposes the running :attr:`handle` for live control
    (pause / resume / terminate, world mutation).

    Args:
        device: Compute device as an ``RS_DeviceLocation`` or a string such as
            ``"cuda:0"`` / ``"cpu"``. ``None`` uses RiseOrigin's default device.
    """

    def __init__(
        self, device: Optional[Union["rs.RS_DeviceLocation", str]] = None
    ) -> None:
        self._rise = rs.RiseOrigin() if device is None else rs.RiseOrigin(device)
        self._handle: Optional["rs.RS_WorldHandle"] = None

    def hello(
        self, world: "rs.RSC_World", callback: Optional[Callable] = None
    ) -> "rs.RS_WorldHandle":
        """Say hello to the world.

        Args:
            world: The ``RSC_World`` to simulate.
            callback: Optional per-step callback
                ``(world_name, time, controllers)`` delivering records and
                accepting control signals.

        Returns:
            The running ``RS_WorldHandle``.
        """
        self._handle = self._rise.hello(world, callback)
        return self._handle

    @property
    def handle(self) -> Optional["rs.RS_WorldHandle"]:
        """The running ``RS_WorldHandle``, or ``None`` before :meth:`hello`."""
        return self._handle


class InstanceRemote(Instance):
    """Attach to a rise-origin world in another process or on another machine.

    Mirrors :class:`InstanceLocal`'s surface over a QUIC transport so a viewer
    attaches identically to a remote world.

    Args:
        address: Host/port (or URI) of the remote ``ControllerServer``.
    """

    def __init__(self, address: str) -> None:
        raise NotImplementedError(
            "InstanceRemote (remote attach over QUIC) is not yet supported"
        )
