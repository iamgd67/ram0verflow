"""
Faster subset-sum solver for mining.

Consensus is unchanged: puzzles still come from pow.instance, and every
answer must pass pow.check_one / pow.verify. This module only replaces
the search.

Backends, first one that loads wins:

* ``c`` — gcc -O3 meet-in-the-middle with an open-addressing table
* ``numpy`` — vectorized half-sums + searchsorted
* ``python`` — the reference dict walk in pow.solve_instance
"""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

from . import pow as powfn

_HERE = Path(__file__).resolve().parent
_LIB = None
_BACKEND = "python"


def backend() -> str:
    return _BACKEND


def _try_load_c():
    src = _HERE / "mitm.c"
    lib_path = _HERE / "mitm.so"
    if not src.exists():
        return None
    stale = (not lib_path.exists()) or lib_path.stat().st_mtime < src.stat().st_mtime
    if stale:
        cmd = [
            "gcc",
            "-O3",
            "-march=native",
            "-shared",
            "-fPIC",
            "-o",
            str(lib_path),
            str(src),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
    try:
        dll = ctypes.CDLL(str(lib_path))
    except OSError:
        return None
    dll.rofl_solve_mitm.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_int64,
    ]
    dll.rofl_solve_mitm.restype = ctypes.c_int64
    return dll


def _half_sums_np(nums):
    import numpy as np

    n = int(nums.size)
    out = np.empty(1 << n, dtype=np.int64)
    out[0] = 0
    for i in range(n):
        half = 1 << i
        out[half : 2 * half] = out[:half] + nums[i]
    return out


def _solve_numpy(numbers, target: int):
    import numpy as np

    nums = np.asarray(numbers, dtype=np.int64)
    n = int(nums.size)
    if n <= 0:
        return None
    half = n // 2
    left = _half_sums_np(nums[:half])
    right = _half_sums_np(nums[half:])
    need = int(target) - right
    order = np.argsort(left, kind="stable")
    ls = left[order]
    idx = np.searchsorted(ls, need)
    in_range = idx < ls.size
    safe = np.where(in_range, idx, 0)
    matched = in_range & (ls[safe] == need)
    for rmask in np.flatnonzero(matched):
        lmask = int(order[int(idx[rmask])])
        full = lmask | (int(rmask) << half)
        if full:
            return full
    return None


def _solve_c(numbers, target: int):
    n = len(numbers)
    buf = (ctypes.c_int64 * n)(*numbers)
    found = _LIB.rofl_solve_mitm(buf, n, int(target))
    return int(found) if found else None


def solve_instance(numbers, target: int):
    """Meet in the middle. Returns a subset mask, or None if there is none."""
    if _BACKEND == "c" and len(numbers) <= 40:
        return _solve_c(numbers, target)
    if _BACKEND in ("c", "numpy"):
        try:
            return _solve_numpy(numbers, target)
        except ImportError:
            pass
    return powfn.solve_instance(numbers, target)


def solve_puzzle(header_core: bytes, j: int, start_nonce: int = 0):
    """Grind nonces for puzzle `j` until an instance has a solution."""
    nonce, subset, _tried = _grind(header_core, j, start_nonce, solve_instance)
    return nonce, subset


def _grind(header_core: bytes, j: int, start_nonce: int, solve_fn):
    for nonce in range(start_nonce, powfn.MAX_NONCE):
        numbers, target = powfn.instance(header_core, j, nonce)
        subset = solve_fn(numbers, target)
        if subset is not None:
            return nonce, subset, nonce - start_nonce + 1
    raise RuntimeError(f"no solvable instance for puzzle {j}; this should not happen")


def _puzzle_task(args):
    header_core, j, solver_name = args
    solve_fn = solve_instance if solver_name == "fast" else powfn.solve_instance
    nonce, subset, tried = _grind(header_core, j, 0, solve_fn)
    return j, nonce, subset, tried


def solve_puzzles_iter(header_core: bytes, k: int, workers=None, solver_name: str = "fast"):
    """Yield ``(j, nonce, subset, tried)`` as each of ``k`` puzzles is solved."""
    if k <= 0:
        return
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(int(workers), k))
    tasks = [(header_core, j, solver_name) for j in range(k)]
    if workers == 1:
        for task in tasks:
            yield _puzzle_task(task)
        return
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers) as pool:
        for result in pool.imap_unordered(_puzzle_task, tasks, chunksize=1):
            yield result


def solve_puzzles(header_core: bytes, k: int, workers=None, solver_name: str = "fast"):
    """
    Solve ``k`` independent puzzles, in parallel when ``workers`` > 1.

    Returns ``(solutions, tried, elapsed)`` with solutions in puzzle order.
    """
    solutions = [None] * k
    tried = 0
    started = time.time()
    for j, nonce, subset, ntry in solve_puzzles_iter(
        header_core, k, workers=workers, solver_name=solver_name
    ):
        solutions[j] = (nonce, subset)
        tried += ntry
    return solutions, tried, time.time() - started


def _init_backend():
    global _LIB, _BACKEND
    _LIB = _try_load_c()
    if _LIB is not None:
        _BACKEND = "c"
        return
    try:
        import numpy  # noqa: F401

        _BACKEND = "numpy"
    except ImportError:
        _BACKEND = "python"


_init_backend()
