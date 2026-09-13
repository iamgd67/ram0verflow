/*
 * Meet-in-the-middle subset-sum for ROFL puzzles.
 *
 * Same contract as rofl.pow.solve_instance: split n numbers in half,
 * enumerate 2^(n/2) subset sums, return a non-empty mask or 0.
 *
 * Built as a shared library and called from solver.py. Consensus
 * generation and verification stay in Python.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_N 40
#define MAX_SIDE (1u << 20)

typedef struct {
    int64_t key;
    uint32_t mask;
    uint32_t gen;
} Slot;

static __thread int64_t *sums_buf;
static __thread uint32_t sums_cap;
static __thread Slot *tab_buf;
static __thread uint32_t tab_cap;
static __thread uint32_t epoch = 1;

static int ensure_sums(uint32_t need) {
    if (need <= sums_cap && sums_buf) {
        return 1;
    }
    if (need > MAX_SIDE) {
        return 0;
    }
    int64_t *next = (int64_t *)realloc(sums_buf, (size_t)need * sizeof(int64_t));
    if (!next) {
        return 0;
    }
    sums_buf = next;
    sums_cap = need;
    return 1;
}

static int ensure_tab(uint32_t need) {
    if (need <= tab_cap && tab_buf) {
        return 1;
    }
    Slot *next = (Slot *)realloc(tab_buf, (size_t)need * sizeof(Slot));
    if (!next) {
        return 0;
    }
    /* realloc does not zero the new tail; epoch makes stale gens ignorable,
       but brand-new bytes must not accidentally match the current epoch. */
    if (need > tab_cap) {
        memset(next + tab_cap, 0, (size_t)(need - tab_cap) * sizeof(Slot));
    }
    if (!tab_cap) {
        memset(next, 0, (size_t)need * sizeof(Slot));
    }
    tab_buf = next;
    tab_cap = need;
    return 1;
}

static uint32_t next_pow2(uint32_t x) {
    uint32_t cap = 8;
    while (cap < x) {
        cap <<= 1;
    }
    return cap;
}

static uint32_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return (uint32_t)(z ^ (z >> 31));
}

static void fill_sums(const int64_t *nums, int n, int64_t *sums) {
    sums[0] = 0;
    for (int i = 0; i < n; i++) {
        uint32_t half = 1u << i;
        int64_t a = nums[i];
        for (uint32_t j = 0; j < half; j++) {
            sums[half + j] = sums[j] + a;
        }
    }
}

int64_t rofl_solve_mitm(const int64_t *numbers, int n, int64_t target) {
    if (numbers == NULL || n <= 0 || n > MAX_N) {
        return 0;
    }

    int left_n = n / 2;
    int right_n = n - left_n;
    uint32_t nL = 1u << left_n;
    uint32_t nR = 1u << right_n;
    uint32_t side = nL > nR ? nL : nR;
    uint32_t cap = next_pow2(nL * 2u);

    if (!ensure_sums(side) || !ensure_tab(cap)) {
        return 0;
    }

    epoch++;
    if (epoch == 0) {
        memset(tab_buf, 0, (size_t)tab_cap * sizeof(Slot));
        epoch = 1;
    }

    fill_sums(numbers, left_n, sums_buf);

    uint32_t cap_mask = cap - 1;
    uint32_t cur = epoch;
    for (uint32_t m = 0; m < nL; m++) {
        int64_t s = sums_buf[m];
        if (s > target) {
            continue;
        }
        uint32_t h = mix64((uint64_t)s) & cap_mask;
        for (;;) {
            Slot *slot = &tab_buf[h];
            if (slot->gen != cur) {
                slot->key = s;
                slot->mask = m;
                slot->gen = cur;
                break;
            }
            if (slot->key == s) {
                break; /* keep the first mask, same as the reference */
            }
            h = (h + 1u) & cap_mask;
        }
    }

    fill_sums(numbers + left_n, right_n, sums_buf);

    for (uint32_t rm = 0; rm < nR; rm++) {
        int64_t s = sums_buf[rm];
        if (s > target) {
            continue;
        }
        int64_t need = target - s;
        uint32_t h = mix64((uint64_t)need) & cap_mask;
        for (;;) {
            Slot *slot = &tab_buf[h];
            if (slot->gen != cur) {
                break;
            }
            if (slot->key == need) {
                int64_t full = (int64_t)slot->mask | ((int64_t)rm << left_n);
                if (full != 0) {
                    return full;
                }
                break;
            }
            h = (h + 1u) & cap_mask;
        }
    }
    return 0;
}
