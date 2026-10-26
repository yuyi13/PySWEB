#!/usr/bin/env python3
"""
Script: verify_allocation.py
Objective: Retry transient scheduler lookups while requiring verified allocation limits.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-23
Last updated: 2026-10-23
Inputs: Resource helper, required PBS limits and declared storage mounts.
Outputs: Preserved resource-check attempts and a verified final resource report.
Usage: python verify_allocation.py --helper PATH --output PATH --ncpus N --memory-gib N --jobfs-gib N --storage MOUNT
Dependencies: Python standard library; configured Gadi resource helper.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


def verify(helper, output, *, ncpus, memory_gib, jobfs_gib, storage,
           attempts=4, delay=10):
    """Never substitute guessed resources when scheduler verification is unavailable."""
    if attempts < 1 or delay < 0:
        raise ValueError('Retry attempts must be positive and delay non-negative')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        path = output.with_name(f'{output.stem}.attempt{attempt}.json')
        subprocess.run([sys.executable, '-S', str(helper), str(path)], check=True)
        report = json.loads(path.read_text())
        temporary = output.with_suffix(output.suffix + '.partial')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output)
        if report.get('allocation_verified') and report.get('safe_for_compute'):
            local = report['local_resources']
            required = {'ncpus': ncpus, 'memory_bytes': memory_gib * 1024**3,
                        'jobfs_bytes': jobfs_gib * 1024**3}
            for key, minimum in required.items():
                if local.get(key) is None or local[key] < minimum:
                    raise RuntimeError(f'Verified allocation does not meet required {key}: {minimum}')
            if not set(storage) <= set((report.get('storage') or '').split('+')):
                raise RuntimeError('Verified allocation is missing required storage mounts')
            return report
        if report.get('scheduler_error') != 'TimeoutExpired':
            raise RuntimeError('Allocation verification failed: ' + '; '.join(report.get('reasons', [])))
        if attempt < attempts:
            print(f'Scheduler lookup timed out; retrying allocation verification ({attempt}/{attempts})', flush=True)
            time.sleep(delay)
    raise RuntimeError(f'Allocation remains unverified after {attempts} scheduler lookup attempts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ncpus', required=True, type=int)
    parser.add_argument('--memory-gib', required=True, type=int)
    parser.add_argument('--jobfs-gib', required=True, type=int)
    parser.add_argument('--storage', required=True, nargs='+')
    args = parser.parse_args()
    verify(args.helper, args.output, ncpus=args.ncpus, memory_gib=args.memory_gib,
           jobfs_gib=args.jobfs_gib, storage=args.storage)


if __name__ == '__main__':
    main()
