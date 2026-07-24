# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The create layer's shared worker pool.

One pool serves every algorithm that parallelizes; the editor owns one
per session and hands it to builds through the editor access, so the
worker spin-up cost is paid once. The call surface duplicates
:class:`multiprocessing.pool.Pool` by delegation, so anyone fluent in
the standard pool uses this one unchanged, plus the shared memory
helpers workers use to read big broadcast arrays without copies.

The pool uses the forkserver start method (spawn on Windows) and warms
the fork server at construction, before any compute context can exist
in the parent process, since forking after a device context is set up
is unsafe. Worker processes themselves start lazily on first use.
"""

import multiprocessing
import sys
import weakref
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

__all__ = ["WorkerPool"]


class WorkerPool:
    """The shared worker processes of one editor session.

    Manages the lazily started process pool and the shared memory
    blocks broadcast to it. Work submission mirrors
    ``multiprocessing.pool.Pool``: ``map``, ``starmap``, ``imap``,
    ``imap_unordered``, ``apply``, their async forms, ``close``,
    ``join``, and ``terminate`` behave exactly like the standard ones.
    ``map`` and ``starmap`` return results in task order, which is what
    keeps parallel builds byte-identical to serial ones.

    Args:
        processes: Worker process count; None takes the machine's cpu
            count, the standard pool default.
    """

    def __init__(self, processes: Optional[int] = None) -> None:
        self.processes = processes
        self._pool = None
        self._is_closed = False
        self._shared_blocks = []
        if sys.platform != "win32":
            self._context = multiprocessing.get_context("forkserver")
            from multiprocessing import forkserver

            forkserver.ensure_running()
        else:
            self._context = multiprocessing.get_context("spawn")

    def apply(self, func, args=(), kwds=None):
        """Run one call on a worker and return its result."""
        return self._get_pool().apply(func, args, kwds or {})

    def apply_async(self, func, args=(), kwds=None, callback=None, error_callback=None):
        """Submit one call and return its async result."""
        return self._get_pool().apply_async(
            func, args, kwds or {}, callback, error_callback
        )

    def map(self, func, iterable, chunksize=None):
        """Run the function over the items; results in task order."""
        return self._get_pool().map(func, iterable, chunksize)

    def map_async(
        self, func, iterable, chunksize=None, callback=None, error_callback=None
    ):
        """Submit a map and return its async result."""
        return self._get_pool().map_async(
            func, iterable, chunksize, callback, error_callback
        )

    def starmap(self, func, iterable, chunksize=None):
        """Run the function over argument tuples; results in task order."""
        return self._get_pool().starmap(func, iterable, chunksize)

    def starmap_async(
        self, func, iterable, chunksize=None, callback=None, error_callback=None
    ):
        """Submit a starmap and return its async result."""
        return self._get_pool().starmap_async(
            func, iterable, chunksize, callback, error_callback
        )

    def imap(self, func, iterable, chunksize=1):
        """Return a lazy iterator of results in task order."""
        return self._get_pool().imap(func, iterable, chunksize)

    def imap_unordered(self, func, iterable, chunksize=1):
        """Return a lazy iterator of results in completion order."""
        return self._get_pool().imap_unordered(func, iterable, chunksize)

    def make_shared_array(self, array: np.ndarray) -> Tuple[str, tuple, str]:
        """Copy an array into a named shared block workers read.

        Args:
            array: The array broadcast to workers.

        Returns:
            The (name, shape, dtype text) handle to pass in task
            arguments; workers open it with :meth:`read_shared_array`.
            The block frees when this pool closes or terminates.
        """
        block = shared_memory.SharedMemory(create=True, size=int(array.nbytes))
        view = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)
        view[...] = array
        self._shared_blocks.append(block)
        return block.name, tuple(array.shape), array.dtype.str

    def free_shared_array(self, name: str) -> None:
        """Close and unlink one shared block by its name.

        Args:
            name: The block name from :meth:`make_shared_array`.
        """
        for block in list(self._shared_blocks):
            if block.name != name:
                continue
            self._shared_blocks.remove(block)
            try:
                block.close()
                block.unlink()
            except FileNotFoundError:
                pass
            return

    @staticmethod
    def read_shared_array(handle: Tuple[str, tuple, str]) -> np.ndarray:
        """Return the worker-side view of a shared array handle.

        The returned view keeps its block open for its own lifetime;
        the creating pool owns the block and frees it on close.

        Args:
            handle: The (name, shape, dtype text) triple from
                :meth:`make_shared_array`.
        """
        name, shape, dtype = handle
        block = shared_memory.SharedMemory(name=name)
        array = np.ndarray(shape, dtype=np.dtype(dtype), buffer=block.buf)
        weakref.finalize(array, block.close)
        return array

    def close(self) -> None:
        """Stop accepting work, wait for workers, free shared blocks."""
        self._is_closed = True
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        self._free_shared_blocks()

    def join(self) -> None:
        """Wait for the workers of a closed or terminated pool."""
        if self._pool is not None:
            self._pool.join()

    def terminate(self) -> None:
        """Stop workers immediately and free shared blocks."""
        self._is_closed = True
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        self._free_shared_blocks()

    def _get_pool(self):
        """Return the running pool, starting it on first use."""
        if self._is_closed:
            raise ValueError("Pool not running")
        if self._pool is None:
            self._pool = self._context.Pool(self.processes)
        return self._pool

    def _free_shared_blocks(self) -> None:
        """Close and unlink every shared block this pool created."""
        for block in self._shared_blocks:
            try:
                block.close()
                block.unlink()
            except FileNotFoundError:
                pass
        self._shared_blocks.clear()
