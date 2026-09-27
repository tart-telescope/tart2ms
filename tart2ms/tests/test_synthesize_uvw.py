#
# Unit tests for UVW synthesis.
# synthesize_uvw is compared against reference output generated with the
# previous casacore.measures implementation (uvw_golden_casacore.npz).
#

import logging
import os
import unittest

import numpy as np

from tart2ms.fixvis import (
    baseline_index,
    baseline_uvw,
    dense2sparse_uvw,
    rephase,
    synthesize_uvw,
)

logger = logging.getLogger("tart2ms")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

GOLDEN = os.path.join(os.path.dirname(__file__), "uvw_golden_casacore.npz")

# astropy/erfa vs casacore differ by frame/precession model details only:
# ~1e-4 relative, i.e. well under a millimetre for TART baselines
UVW_TOLERANCE_M = 1e-3


class TestSynthesizeUVW(unittest.TestCase):
    """Test synthesize_uvw against the casacore reference implementation."""

    @classmethod
    def setUpClass(cls):
        cls.golden = np.load(GOLDEN)

    def _inputs(self, tag):
        g = self.golden
        return dict(
            station_ECEF=g[f"{tag}_station"],
            time=g[f"{tag}_time"],
            a1=g[f"{tag}_a1"],
            a2=g[f"{tag}_a2"],
            phase_ref=g[f"{tag}_phase_ref"],
        )

    def _check_golden(self, tag):
        result = synthesize_uvw(ack=False, **self._inputs(tag))
        max_diff = np.max(np.abs(result["UVW"] - self.golden[f"{tag}_uvw"]))
        self.assertLess(max_diff, UVW_TOLERANCE_M, f"UVW mismatch: max diff = {max_diff}")
        return result

    def test_matches_casacore_small(self):
        inputs = self._inputs("small")
        result = self._check_golden("small")
        na = 8
        nbl = na * (na - 1) // 2 + na
        unique_time = np.unique(inputs["time"])
        np.testing.assert_array_equal(result["TIME_CENTROID"], unique_time.repeat(nbl))
        a1, a2 = np.triu_indices(na, 0)
        np.testing.assert_array_equal(result["ANTENNA1"], np.tile(a1, unique_time.size))
        np.testing.assert_array_equal(result["ANTENNA2"], np.tile(a2, unique_time.size))

    def test_matches_casacore_large(self):
        self._check_golden("large")

    def test_baseline_length_preserved(self):
        """UVW is a rotation of the ITRF baseline: lengths must be preserved."""
        inputs = self._inputs("large")
        result = synthesize_uvw(ack=False, **inputs)
        st = inputs["station_ECEF"]
        bl = st[result["ANTENNA1"]] - st[result["ANTENNA2"]]
        np.testing.assert_allclose(
            np.linalg.norm(result["UVW"], axis=1), np.linalg.norm(bl, axis=1), atol=1e-9
        )

    def test_baseline_uvw_matches_dense_lookup(self):
        """Per-row baseline_uvw equals synthesize_uvw + dense2sparse_uvw."""
        inputs = self._inputs("small")
        rng = np.random.default_rng(1)
        nrow = 60
        time = rng.choice(np.unique(inputs["time"]), nrow)
        a1 = rng.integers(0, 8, nrow)
        a2 = rng.integers(0, 8, nrow)
        padded = synthesize_uvw(
            station_ECEF=inputs["station_ECEF"],
            time=time,
            a1=a1,
            a2=a2,
            phase_ref=inputs["phase_ref"],
            ack=False,
        )
        expected = dense2sparse_uvw(
            a1, a2, time, np.zeros(nrow, dtype=int), padded["UVW"], ack=False
        )
        rows = baseline_uvw(inputs["station_ECEF"], time, a1, a2, inputs["phase_ref"])
        np.testing.assert_allclose(rows, expected, atol=1e-12)

    def test_per_row_phase_directions(self):
        """dir_index selects a phase centre per row."""
        inputs = self._inputs("small")
        dirs = np.array([[0.5, 0.52359878], inputs["phase_ref"][0]])
        dir_index = np.arange(inputs["time"].size) % 2
        rows = baseline_uvw(
            inputs["station_ECEF"], inputs["time"], inputs["a1"], (inputs["a2"] + 1) % 8,
            dirs, dir_index=dir_index,
        )
        for d in range(2):
            sel = dir_index == d
            single = baseline_uvw(
                inputs["station_ECEF"], inputs["time"][sel], inputs["a1"][sel],
                (inputs["a2"][sel] + 1) % 8, dirs[d:d + 1],
            )
            np.testing.assert_allclose(rows[sel], single, atol=1e-12)


class TestRephase(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.nrow = 12
        self.vis = (rng.normal(size=(self.nrow, 1, 2))
                    + 1j * rng.normal(size=(self.nrow, 1, 2))).astype(np.complex64)
        self.uvw = rng.normal(size=(self.nrow, 3))
        self.field_ids = np.arange(self.nrow) % 3
        self.refdir = np.array([[10.0, -20.0], [11.0, -21.0], [12.0, -22.0]])
        self.freq = np.array([1.57542e9])

    def test_identity_when_centre_unchanged(self):
        sel = np.ones(self.nrow, dtype=bool)
        out = rephase(self.vis, self.uvw, self.field_ids, sel, self.freq,
                      self.refdir, self.refdir)
        np.testing.assert_allclose(out, self.vis, rtol=1e-6)

    def test_single_centre_matches_per_field_centres(self):
        sel = np.ones(self.nrow, dtype=bool)
        pos = np.array([30.0, -45.0])
        single = rephase(self.vis, self.uvw, self.field_ids, sel, self.freq, pos, self.refdir)
        per_field = rephase(self.vis, self.uvw, self.field_ids, sel, self.freq,
                            np.tile(pos, (3, 1)), self.refdir)
        np.testing.assert_allclose(single, per_field)
        self.assertEqual(np.abs(single).shape, self.vis.shape)
        np.testing.assert_allclose(np.abs(single), np.abs(self.vis), rtol=1e-5)

    def test_unselected_rows_are_zero(self):
        sel = self.field_ids != 1
        out = rephase(self.vis, self.uvw, self.field_ids, sel, self.freq,
                      np.array([30.0, -45.0]), self.refdir)
        self.assertTrue(np.all(out[~sel] == 0))


class TestDense2SparseUVW(unittest.TestCase):
    """Test that the vectorized dense2sparse_uvw matches the original."""

    def _dense2sparse_original(self, a1, a2, time, ddid, padded_uvw):
        """Original per-row loop for comparison."""
        assert time.size == a1.size
        assert a1.size == a2.size
        ants = np.concatenate((a1, a2))
        unique_ants = np.arange(np.max(ants) + 1)
        na = unique_ants.size
        nbl = na * (na - 1) // 2 + na
        unique_time = np.unique(time)
        new_uvw = np.zeros((a1.size, 3), dtype=padded_uvw.dtype)
        outbl = baseline_index(a1, a2, na)
        for outrow in range(a1.size):
            lookupt = np.argwhere(unique_time == time[outrow])
            new_uvw[outrow][:] = padded_uvw[lookupt * nbl + outbl[outrow], :]
        return new_uvw

    def test_matches_original(self):
        """Vectorized version must match per-row loop exactly."""
        np.random.seed(123)
        na = 8
        nbl = na * (na - 1) // 2 + na
        ntime = 10

        padded_uvw = np.arange(ntime * nbl * 3, dtype=np.float64).reshape(ntime * nbl, 3)
        padded_uvw += np.random.randn(ntime * nbl, 3) * 0.01

        unique_time = np.linspace(5071671511.0, 5071672000.0, ntime)
        nrows = ntime * (nbl - 3)
        time = np.repeat(unique_time, nbl - 3)
        # Build a1/a2 for all baselines except the last 3
        tris = np.stack(np.triu_indices(na, 0), axis=1)
        a1 = np.tile(tris[:-3, 0], ntime)
        a2 = np.tile(tris[:-3, 1], ntime)
        ddid = np.zeros(nrows, dtype=int)

        result_opt = dense2sparse_uvw(a1, a2, time, ddid, padded_uvw)
        result_orig = self._dense2sparse_original(a1, a2, time, ddid, padded_uvw)

        np.testing.assert_array_equal(result_opt, result_orig)

    def test_full_baseline_set(self):
        """Test with all baselines present."""
        np.random.seed(456)
        na = 6
        nbl = na * (na - 1) // 2 + na
        ntime = 5

        padded_uvw = np.random.randn(ntime * nbl, 3).astype(np.float64)

        tris = np.stack(np.triu_indices(na, 0), axis=1)
        unique_time = np.linspace(5071671511.0, 5071671800.0, ntime)
        time = np.repeat(unique_time, nbl)
        a1 = np.tile(tris[:, 0], ntime)
        a2 = np.tile(tris[:, 1], ntime)
        ddid = np.zeros(ntime * nbl, dtype=int)

        result_opt = dense2sparse_uvw(a1, a2, time, ddid, padded_uvw)
        result_orig = self._dense2sparse_original(a1, a2, time, ddid, padded_uvw)

        np.testing.assert_array_equal(result_opt, result_orig)
        np.testing.assert_array_equal(result_opt, padded_uvw)

    def test_single_baseline(self):
        """Test with single baseline across many timestamps."""
        na = 4
        nbl = na * (na - 1) // 2 + na
        ntime = 100

        padded_uvw = np.random.randn(ntime * nbl, 3).astype(np.float64)

        unique_time = np.linspace(5071671511.0, 5071673000.0, ntime)
        time = unique_time.copy()
        a1 = np.zeros(ntime, dtype=int)
        a2 = np.ones(ntime, dtype=int)
        ddid = np.zeros(ntime, dtype=int)

        result_opt = dense2sparse_uvw(a1, a2, time, ddid, padded_uvw)
        result_orig = self._dense2sparse_original(a1, a2, time, ddid, padded_uvw)

        np.testing.assert_array_equal(result_opt, result_orig)

    def test_large_realistic(self):
        """Test with realistic TART-scale: 24 ants, 100 timestamps."""
        na = 24
        nbl = na * (na - 1) // 2 + na
        ntime = 100

        np.random.seed(789)
        padded_uvw = np.random.randn(ntime * nbl, 3).astype(np.float64)

        tris = np.stack(np.triu_indices(na, 0), axis=1)
        unique_time = np.linspace(5071671511.0, 5071672000.0, ntime)
        time = np.repeat(unique_time, nbl)
        a1 = np.tile(tris[:, 0], ntime)
        a2 = np.tile(tris[:, 1], ntime)
        ddid = np.zeros(ntime * nbl, dtype=int)

        result_opt = dense2sparse_uvw(a1, a2, time, ddid, padded_uvw)
        result_orig = self._dense2sparse_original(a1, a2, time, ddid, padded_uvw)

        np.testing.assert_array_equal(result_opt, result_orig)


if __name__ == "__main__":
    unittest.main()
