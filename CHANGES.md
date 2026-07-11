# Changes

## Unreleased (post v0.8.1)

### Performance optimizations for `--rephase obs-midpoint`

#### `synthesize_uvw`: vectorize per-antenna inner loop (commit `57c7c47`)

**Problem**: The inner loop over antennas (lines ~193-202 of `fixvis.py`) constructed a new casacore `dm.baseline()` measure and called `dm.to_uvw()` for every antenna at every timestamp — O(na × ntime) calls for both. For a typical 24-antenna × 1000-timestamp TART dataset this meant 24,000 baseline constructions and 24,000 `to_uvw` calls.

**Fix**: Pre-build a single vectorized `dm.baseline()` measure containing all antenna pairs with the reference antenna, using casacore's native vector quantity support. Then call `dm.to_uvw(baseline_measure)` once per timestamp. The measure is re-evaluated correctly in each new epoch frame.

**Speedup**: `dm.baseline()` calls drop from na × ntime to 1. `dm.to_uvw()` calls drop from na × ntime to ntime. A ~24× reduction in casacore measure operations for a 24-antenna array.

**Testing**: 5 new tests in `tart2ms/tests/test_synthesize_uvw.py` (`TestSynthesizeUVW`) verify bit-identical output against the original per-antenna loop across small and large arrays (4-24 antennas, 1-50 timestamps), different phase reference directions, and sparse antenna configurations.

---

#### `dense2sparse_uvw`: vectorize per-row Python loop (commit `f9587b3`)

**Problem**: A Python `for` loop iterated over every row calling `np.argwhere(unique_time == time[outrow])` — O(rows²) lookups. For 30,000 rows (100 timestamps × 300 baselines) this was ~0.15s per call, and it's called twice per MS creation (once in `ms_create`, once in `fixms`).

**Fix**: Build a `time → index` dict from `unique_time` once, compute all flat indices with integer array math (`time_indices * nbl + outbl`), then do a single numpy fancy-index assignment: `new_uvw[:] = padded_uvw[flat_idx, :]`.

**Speedup**: ~53× for 30,000 rows. Per-call time drops from ~0.15s to ~0.003s.

**Testing**: 4 new tests in `tart2ms/tests/test_synthesize_uvw.py` (`TestDense2SparseUVW`) verify bit-identical output against the original per-row loop across sparse baselines, full baseline sets, single-baseline, and realistic 24-antenna × 100-timestamp configurations.

---

### v0.8.1 — Performance optimizations and daskms compatibility fix

**Bugs fixed**:

- **daskms Dataset infinite recursion** (`0bcbe12`): daskms 0.2.32 with dask ≥2024.x can infinite-recurse in `Dataset.__getattr__` when the multiprocessing scheduler introspects the object during serialization. Patched `Dataset.__getattr__` to raise `AttributeError` immediately for dunder-named attributes, short-circuiting the recursion.

**Performance**:

- **Move `.compute()` calls out of UVW subfield loop** (`0bcbe12`): `map_row_to_zendir.compute()` and `antenna_itrf_pos.compute()` were called inside the per-subfield loop body. Hoisted to before the loop; results reused.

- **Numpy broadcasting in `rephase()`** (`0bcbe12`): Replaced 3 explicit `.repeat()/.reshape()` array copies and `np.tile` with broadcasting in the phase factor computation. Eliminates `(nrowsel, nfreq, 3)` intermediate array allocations.

- **Default chunk size increased** (`0bcbe12`): Changed from 10,000 to 100,000 rows. Exposed as module-level `DEFAULT_CHUNK_SIZE` constant, used consistently across `ms_create`, `ms_from_hdf5`, `ms_from_json`, and the CLI script.
