#!/usr/bin/env python3
"""
Fast subset-sum solver tests.

The consensus path (instance / check_one / verify) stays in rofl.pow.
This suite only checks that the faster miner solver returns answers
those functions accept, including when N is patched the way test_chain.py
does.

    python3 tests/test_solver.py
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rofl import pow as powfn  # noqa: E402


CORE = b"rofl-fast-solver-test|height=1|miner=bench"


def _instances(core, count, start_nonce=0):
    """Yield (nonce, numbers, target) for `count` consecutive nonces."""
    for nonce in range(start_nonce, start_nonce + count):
        numbers, target = powfn.instance(core, 0, nonce)
        yield nonce, numbers, target


class FastSolverContract(unittest.TestCase):
    def test_module_exports_solver_api(self):
        from rofl import solver

        self.assertTrue(callable(solver.solve_instance))
        self.assertTrue(callable(solver.solve_puzzle))
        self.assertTrue(callable(solver.solve_puzzles))
        self.assertTrue(callable(solver.backend))

    def test_agrees_with_reference_on_solvable_and_unsolvable(self):
        from rofl import solver

        checked = 0
        found = 0
        for nonce, numbers, target in _instances(CORE, 12):
            ref = powfn.solve_instance(numbers, target)
            fast = solver.solve_instance(numbers, target)
            if ref is None:
                self.assertIsNone(fast, f"nonce {nonce} is unsolvable")
            else:
                self.assertIsNotNone(fast, f"nonce {nonce} is solvable")
                self.assertTrue(
                    powfn.check_one(CORE, 0, nonce, fast),
                    f"fast mask {fast} failed check_one at nonce {nonce}",
                )
                found += 1
            checked += 1
        self.assertEqual(checked, 12)
        self.assertGreaterEqual(found, 3)

    def test_empty_subset_is_rejected(self):
        from rofl import solver

        numbers = [1] * powfn.N
        self.assertIsNone(solver.solve_instance(numbers, 0))

    def test_singleton_subset_is_found(self):
        from rofl import solver

        numbers = [1 << i for i in range(powfn.N)]
        target = numbers[7]
        mask = solver.solve_instance(numbers, target)
        self.assertEqual(mask, 1 << 7)

    def test_solve_puzzle_verifies(self):
        from rofl import solver

        nonce, subset = solver.solve_puzzle(CORE, 3)
        self.assertTrue(powfn.check_one(CORE, 3, nonce, subset))
        self.assertTrue(0 <= nonce < powfn.MAX_NONCE)

    def test_parallel_puzzles_all_verify(self):
        from rofl import solver

        k = 4
        solutions, tried, elapsed = solver.solve_puzzles(CORE, k, workers=2)
        self.assertEqual(len(solutions), k)
        self.assertGreaterEqual(tried, k)
        self.assertGreater(elapsed, 0)
        self.assertTrue(powfn.verify(CORE, powfn.encode_solutions(solutions), k))

    def test_respects_patched_n_like_test_chain(self):
        from rofl import solver

        old = powfn.N, powfn.B, powfn.UNIT_WORK
        try:
            powfn.N = 16
            powfn.B = powfn.N - 2
            powfn.UNIT_WORK = 1 << (powfn.N // 2)
            numbers, target = powfn.instance(CORE, 1, 0)
            self.assertEqual(len(numbers), 16)
            mask = solver.solve_instance(numbers, target)
            if mask is not None:
                self.assertTrue(powfn.check_one(CORE, 1, 0, mask))
            nonce, subset = solver.solve_puzzle(CORE, 1)
            self.assertTrue(powfn.check_one(CORE, 1, nonce, subset))
        finally:
            powfn.N, powfn.B, powfn.UNIT_WORK = old


class FastSolverSpeed(unittest.TestCase):
    def test_single_instance_at_least_2x_reference(self):
        from rofl import solver

        batch = list(_instances(CORE + b"|speed", 6, start_nonce=1))
        # Warm both paths so the first hash-table / numpy alloc is not timed.
        powfn.solve_instance(*batch[0][1:])
        solver.solve_instance(*batch[0][1:])

        t0 = time.perf_counter()
        for _, numbers, target in batch:
            powfn.solve_instance(numbers, target)
        ref_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _, numbers, target in batch:
            solver.solve_instance(numbers, target)
        fast_s = time.perf_counter() - t0

        speedup = ref_s / fast_s if fast_s > 0 else float("inf")
        print(
            f"\n  speed  ref {ref_s:.3f}s  fast {fast_s:.3f}s  "
            f"{speedup:.2f}x  backend={solver.backend()}",
            flush=True,
        )
        self.assertGreaterEqual(
            speedup,
            2.0,
            f"expected >=2x, got {speedup:.2f}x (ref {ref_s:.3f}s, fast {fast_s:.3f}s)",
        )


class MinerEntry(unittest.TestCase):
    def test_mine_default_uses_fast_solver(self):
        import miner
        from rofl.consensus import Block

        block = Block(
            height=1,
            prev_hash="00" * 32,
            merkle_root="11" * 32,
            timestamp=1_750_000_000,
            bits=0x1F010000,
            miner="fast-miner",
        )
        solutions, tried, elapsed = miner.mine(block, 2, solver_name="fast", workers=2)
        self.assertEqual(len(solutions), 2)
        self.assertGreaterEqual(tried, 2)
        self.assertGreater(elapsed, 0)
        self.assertTrue(
            powfn.verify(block.header_core(), powfn.encode_solutions(solutions), 2)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
