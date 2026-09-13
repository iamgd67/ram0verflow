#!/usr/bin/env python3
"""
Compare the reference meet-in-the-middle against the fast miner solver
on the same batch of n=40 instances.

    python3 bench_solver.py
    python3 bench_solver.py --count 8 --out bench_results.txt
"""

from __future__ import annotations

import argparse
import time

from rofl import pow as powfn
from rofl import solver as fast_solver


CORE = b"rofl-bench|v1|cpu-solver"


def make_batch(core: bytes, count: int, start_nonce: int = 0):
    batch = []
    for nonce in range(start_nonce, start_nonce + count):
        numbers, target = powfn.instance(core, 0, nonce)
        batch.append((nonce, numbers, target))
    return batch


def time_batch(solve_fn, batch):
    per = []
    t0 = time.perf_counter()
    for nonce, numbers, target in batch:
        one = time.perf_counter()
        mask = solve_fn(numbers, target)
        per.append((nonce, time.perf_counter() - one, mask is not None))
    return time.perf_counter() - t0, per


def main():
    ap = argparse.ArgumentParser(description="Benchmark ROFL subset-sum solvers.")
    ap.add_argument("--count", type=int, default=8, help="instances to solve")
    ap.add_argument("--workers", type=int, default=None, help="parallel puzzles for the k-demo")
    ap.add_argument("--puzzles", type=int, default=8, help="parallel puzzle count")
    ap.add_argument("--out", default="bench_results.txt", help="write a copy of the report")
    args = ap.parse_args()

    batch = make_batch(CORE, args.count)
    # Warmup so the first C/numpy allocation is not in the timed window.
    powfn.solve_instance(*batch[0][1:])
    fast_solver.solve_instance(*batch[0][1:])

    ref_total, ref_per = time_batch(powfn.solve_instance, batch)
    fast_total, fast_per = time_batch(fast_solver.solve_instance, batch)

    lines = []
    def emit(msg=""):
        lines.append(msg)
        print(msg)

    emit(f"ROFL solver benchmark")
    emit(f"  n={powfn.N}  instances={args.count}  fast backend={fast_solver.backend()}")
    emit()
    emit(f"{'nonce':>8}  {'ref_s':>10}  {'fast_s':>10}  {'speedup':>8}  solvable")
    emit("-" * 56)
    speedups = []
    for (nonce, ref_s, ref_ok), (_, fast_s, fast_ok) in zip(ref_per, fast_per):
        if ref_ok != fast_ok:
            raise SystemExit(f"disagreement at nonce {nonce}: ref={ref_ok} fast={fast_ok}")
        ratio = ref_s / fast_s if fast_s > 0 else float("inf")
        speedups.append(ratio)
        emit(
            f"{nonce:8d}  {ref_s:10.4f}  {fast_s:10.4f}  {ratio:7.2f}x  "
            f"{'yes' if ref_ok else 'no'}"
        )
    emit("-" * 56)
    overall = ref_total / fast_total if fast_total > 0 else float("inf")
    emit(
        f"{'total':>8}  {ref_total:10.4f}  {fast_total:10.4f}  {overall:7.2f}x  "
        f"mean per-instance {sum(speedups)/len(speedups):.2f}x"
    )
    emit()

    k = args.puzzles
    workers = args.workers if args.workers is not None else (os_cpu())
    t0 = time.perf_counter()
    solutions, tried, elapsed = fast_solver.solve_puzzles(CORE, k, workers=workers)
    wall = time.perf_counter() - t0
    blob = powfn.encode_solutions(solutions)
    if not powfn.verify(CORE, blob, k):
        raise SystemExit("parallel puzzle solutions failed verify")
    emit(
        f"parallel  k={k} workers={workers}  wall {wall:.3f}s  "
        f"reported {elapsed:.3f}s  tried {tried}  "
        f"{wall/k:.3f}s per puzzle"
    )
    emit()
    emit("answers pass pow.check_one / pow.verify")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def os_cpu():
    import os

    return os.cpu_count() or 1


if __name__ == "__main__":
    main()
