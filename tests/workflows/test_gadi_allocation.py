#!/usr/bin/env python3
"""
Script: test_gadi_allocation.py
Objective: Verify bounded scheduler retries and rejection of insufficient allocations.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-23
Last updated: 2026-10-23
Inputs: Small synthetic scheduler reports; no PBS submissions or scientific data.
Outputs: Unit assertions for resource verification and retained retry evidence.
Usage: python tests/workflows/test_gadi_allocation.py
Dependencies: Python standard library.
"""
from pathlib import Path
from unittest.mock import patch
import importlib.util
import json
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[2] / 'workflows/hpc/gadi/verify_allocation.py'
SPEC = importlib.util.spec_from_file_location('verify_allocation', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AllocationTests(unittest.TestCase):
    def invoke(self, reports):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output = Path(directory.name) / 'resources.json'
        remaining = iter(reports)

        def run(command, check):
            Path(command[-1]).write_text(json.dumps(next(remaining)))

        with patch.object(MODULE.subprocess, 'run', side_effect=run) as process:
            MODULE.verify('helper.py', output, ncpus=1, memory_gib=4,
                          jobfs_gib=1, storage=['scratch/dt1'], attempts=3, delay=0)
        return output, process.call_count

    @staticmethod
    def valid():
        return dict(allocation_verified=True, safe_for_compute=True,
                    local_resources=dict(ncpus=1, memory_bytes=4 * 1024**3,
                                         jobfs_bytes=1024**3), storage='scratch/dt1')

    def test_timeout_then_verified_preserves_attempts(self):
        path, calls = self.invoke([dict(scheduler_error='TimeoutExpired'), self.valid()])
        self.assertEqual(calls, 2)
        self.assertEqual(len(list(path.parent.glob('resources.attempt*.json'))), 2)
        self.assertTrue(json.loads(path.read_text())['allocation_verified'])

    def test_repeated_timeout_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'after 3'):
            self.invoke([dict(scheduler_error='TimeoutExpired')] * 3)

    def test_real_verification_failure_is_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, 'wrong owner'):
            self.invoke([dict(reasons=['wrong owner'])])

    def test_verified_but_insufficient_memory_rejected(self):
        report = self.valid()
        report['local_resources']['memory_bytes'] = 1024**3
        with self.assertRaisesRegex(RuntimeError, 'memory_bytes'):
            self.invoke([report])

    def test_verified_but_missing_mount_rejected(self):
        report = self.valid()
        report['storage'] = 'gdata/ym05'
        with self.assertRaisesRegex(RuntimeError, 'storage mounts'):
            self.invoke([report])


if __name__ == '__main__':
    unittest.main()
