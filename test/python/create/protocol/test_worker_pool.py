# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the shared worker pool: mapping, shared arrays, closing.

Worker functions must come from importable modules, never from this
test module: pytest imports tests through its own importer, so a
worker cannot unpickle anything defined here.
"""

import math

import numpy as np

from rise_origin.create.protocol.worker_pool import WorkerPool


class TestWorkerPool:
    def test_map_returns_results_in_task_order(self):
        pool = WorkerPool(2)
        try:
            values = [float(k * k) for k in range(20)]
            assert pool.map(math.sqrt, values) == [float(k) for k in range(20)]
        finally:
            pool.close()

    def test_shared_array_reads_back_on_workers(self):
        pool = WorkerPool(2)
        try:
            array = np.arange(1000, dtype=np.int64)
            handle = pool.make_shared_array(array)
            reads = pool.map(WorkerPool.read_shared_array, [handle] * 3)
            for read in reads:
                assert (read == array).all()
            pool.free_shared_array(handle[0])
        finally:
            pool.close()

    def test_closed_pool_refuses_work(self):
        pool = WorkerPool(1)
        pool.close()
        try:
            pool.map(math.sqrt, [1.0])
        except ValueError:
            return
        raise AssertionError("closed pool accepted work")
