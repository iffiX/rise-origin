# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The create layer's logger, the one handed to module bodies.

The editor access hands every module body a :class:`Logger` named by
the four-field module name under creation, so every line reads
``rise_origin.nature.tree.willow: <event>`` without the module naming
itself. Messages flow through the standard :mod:`logging` tree, so
filtering by name subtree and custom handlers work the standard way;
the package installs a null handler and the editor attaches a console
handler per session unless told not to.
"""

import logging
import time
from typing import Optional

__all__ = ["Logger", "LoggerProgress"]

logging.getLogger("rise_origin").addHandler(logging.NullHandler())


class Logger:
    """One named logger of the create layer.

    Wraps the standard logger of the given name; the level methods
    delegate unchanged, and :meth:`make_progress` adds throttled
    progress reporting for long stages.

    Args:
        name: The logger name, a four-field module name for module
            builds or the session name for editor events.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._logger = logging.getLogger(name)

    def debug(self, message: str, *args) -> None:
        """Log a debug line."""
        self._logger.debug(message, *args)

    def info(self, message: str, *args) -> None:
        """Log an info line."""
        self._logger.info(message, *args)

    def warning(self, message: str, *args) -> None:
        """Log a warning line."""
        self._logger.warning(message, *args)

    def error(self, message: str, *args) -> None:
        """Log an error line."""
        self._logger.error(message, *args)

    def make_progress(
        self, event: str, total: Optional[int] = None
    ) -> "LoggerProgress":
        """Return a progress reporter for one long stage.

        Args:
            event: The stage name progress lines carry.
            total: The step count the stage completes at; None reports
                bare counts.
        """
        return LoggerProgress(self, event, total)


class LoggerProgress:
    """Throttled progress reporting of one stage through a logger.

    Steps accumulate silently and log an info line at most once per
    interval and on completion, so a tight loop can step every item
    without flooding.

    Args:
        logger: The logger progress lines flow through.
        event: The stage name every line carries.
        total: The step count the stage completes at; None reports
            bare counts.
        min_interval_s: Fewest seconds between two progress lines.
    """

    def __init__(
        self,
        logger: Logger,
        event: str,
        total: Optional[int] = None,
        min_interval_s: float = 2.0,
    ) -> None:
        self.event = event
        self.total = total
        self.count = 0
        self._logger = logger
        self._min_interval_s = min_interval_s
        self._started = time.perf_counter()
        self._last_logged = self._started

    def step(self, count: int = 1) -> None:
        """Advance the stage and log when due.

        Args:
            count: Steps to advance by.
        """
        self.count += count
        now = time.perf_counter()
        is_complete = self.total is not None and self.count >= self.total
        if not is_complete and now - self._last_logged < self._min_interval_s:
            return
        self._last_logged = now
        if self.total:
            share = 100.0 * self.count / self.total
            self._logger.info(
                "%s %d/%d (%.0f%%)", self.event, self.count, self.total, share
            )
        else:
            self._logger.info("%s %d", self.event, self.count)

    def finish(self) -> None:
        """Log the stage's final count and elapsed time."""
        elapsed = time.perf_counter() - self._started
        self._logger.info("%s done: %d in %.1fs", self.event, self.count, elapsed)
