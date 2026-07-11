# Changes

## v0.9.0 — Performance overhaul

### Performance optimizations

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

#### `predict_model`: hoist `.compute()` and `np.array()` out of per-timestamp loop (commit `36970df`)

**Problem**: `map_row_to_zendir.compute()` and `np.array(epoch_s_sources)` were called inside the per-timestamp loop. Since `predict_model` can be invoked up to 3 times (GNSS + celestial + solar catalogs), this meant up to 3N redundant dask `.compute()` and numpy array constructions for N timestamps.

**Fix**: Pre-compute both values once before the loop: `map_row_to_zendir_np = map_row_to_zendir.compute()` and `epoch_s_sources_arr = np.array(epoch_s_sources)`. Reuse inside the loop.

**Testing**: 3 new tests in `tart2ms/tests/test_predict_model.py` (`TestPredictModel`) verify identical output against a reference implementation that keeps the original per-iteration behavior, across single timestamp, multi-timestamp with shared catalog epoch, and multi-timestamp scenarios.

---

#### SOURCE table: avoid per-source dask arrays (commit `fe3647e`)

**Problem**: Each GNSS source created its own single-element dask array (`da.asarray(np.asarray([1]))`), and `da.concatenate()` joined hundreds of them into a deep dask graph. This added unnecessary graph complexity and serialization overhead during `dask.compute()`.

**Fix**: Collect values into plain Python lists during iteration, convert to contiguous numpy arrays once at the end, then wrap each column in a single `da.from_array()` call.

**Testing**: 4 new tests in `tart2ms/tests/test_source_table.py` (`TestSourceTable`) verify identical NUM_LINES, NAME, TIME, and DIRECTION output against the old per-source dask approach, plus shape consistency and column dimensionality checks.

---

### GNSS source catalog: integrate tart-catalogue-client

#### `celestial_positions` for direct RA/Dec (commits `e41070a`, `877c741`, `a1b4e25`)

**Problem**: The legacy per-timestamp REST API made N HTTP requests for N timestamps. Downloaded Az/El positions, then `predict_model` converted them back to J2000 RA/Dec via `azel2radec()` — a wasteful roundtrip.

**Fix**: `tart-catalogue-client` downloads TLE ephemerides once, caches on disk (12-hour TTL), and propagates SGP4 locally for any timestamp. `celestial_positions()` returns J2000 RA/Dec directly, skipping the Az/El roundtrip. `predict_model` detects `ra`/`dec` keys and uses them directly, falling back to `azel2radec` for legacy sources. Legacy REST API path removed entirely.

**Testing**: 5 tests in `tart2ms/tests/test_fetch_sources.py` verify client path output, elevation/name filtering, RA/Dec range validation, and end-to-end integration.

---

### Bug fixes

- **IERS auto_max_age** (`1a47b59`): Astropy coordinate transforms fail when IERS tables are >30 days old. Set `iers.conf.auto_max_age = None` at module level. Also fixed in `tart-catalogue-client` v0.5.1.

- **SOURCE table DIRECTION shape** (`1a47b59`): Guard against empty source lists producing 1D `(0,)` arrays instead of required 2D `(0, 2)`. Use `max(n_src, 1)` for chunk sizes to avoid zero-sized dask chunks.

---

### v0.8.1 — Performance optimizations and daskms compatibility fix

**Bugs fixed**:

- **daskms Dataset infinite recursion** (`0bcbe12`): daskms 0.2.32 with dask ≥2024.x can infinite-recurse in `Dataset.__getattr__` when the multiprocessing scheduler introspects the object during serialization. Patched `Dataset.__getattr__` to raise `AttributeError` immediately for dunder-named attributes, short-circuiting the recursion.

**Performance**:

- **Move `.compute()` calls out of UVW subfield loop** (`0bcbe12`): `map_row_to_zendir.compute()` and `antenna_itrf_pos.compute()` were called inside the per-subfield loop body. Hoisted to before the loop; results reused.

- **Numpy broadcasting in `rephase()`** (`0bcbe12`): Replaced 3 explicit `.repeat()/.reshape()` array copies and `np.tile` with broadcasting in the phase factor computation. Eliminates `(nrowsel, nfreq, 3)` intermediate array allocations.

- **Default chunk size increased** (`0bcbe12`): Changed from 10,000 to 100,000 rows. Exposed as module-level `DEFAULT_CHUNK_SIZE` constant, used consistently across `ms_create`, `ms_from_hdf5`, `ms_from_json`, and the CLI script.
