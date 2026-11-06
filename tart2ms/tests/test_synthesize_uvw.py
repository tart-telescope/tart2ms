#
# Unit test for the vectorized synthesize_uvw optimization.
# Compares output of the optimized function against a per-antenna loop
# implementation to ensure correctness, and against golden values captured
# from the old python-casacore (pyrap) implementation before it was removed.
#

import unittest
import logging

import numpy as np
from astropy import units as u
from astropy.time import Time

from tart2ms.fixvis import (
    _TIME_SCALES,
    _itrf_uvw_frame,
    baseline_index,
    dense2sparse_uvw,
    synthesize_uvw,
)

logger = logging.getLogger("tart2ms")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


# Reference implementation: the original per-antenna loop version
def _synthesize_uvw_original(
    station_ECEF,
    time,
    a1,
    a2,
    phase_ref,
    stopctr_units=["rad", "rad"],
    stopctr_epoch="j2000",
    time_TZ="UTC",
    time_unit="s",
    posframe="ITRF",
    posunits=["m", "m", "m"],
):
    """Loop reference implementation for comparison testing.

    Identical algorithm to :func:`tart2ms.fixvis.synthesize_uvw` but with
    explicit per-antenna and per-baseline Python loops instead of the
    vectorized numpy operations, so the test isolates the vectorization.
    """
    assert time.size == a1.size
    assert a1.size == a2.size

    ants = np.concatenate((a1, a2))
    unique_ants = np.arange(np.max(ants) + 1)
    unique_time = np.unique(time)
    na = unique_ants.size
    nbl = na * (na - 1) // 2 + na
    ntime = unique_time.size

    padded_uvw = np.zeros((ntime * nbl, 3), dtype=np.float64)
    antindices = np.stack(np.triu_indices(na, 0), axis=1)
    padded_time = unique_time.repeat(nbl)
    padded_a1 = np.tile(antindices[:, 0], (1, ntime)).ravel()
    padded_a2 = np.tile(antindices[:, 1], (1, ntime)).ravel()

    ra = u.Quantity(phase_ref[0, 0], stopctr_units[0]).to_value(u.rad)
    dec = u.Quantity(phase_ref[0, 1], stopctr_units[1]).to_value(u.rad)
    scale = _TIME_SCALES.get(str(time_TZ).upper(), "utc")

    for ti, t in enumerate(unique_time):
        mjd = u.Quantity(t, time_unit).to_value(u.day)
        if scale != "utc":
            mjd = Time(mjd, format="mjd", scale=scale).utc.mjd
        u_vec, v_vec, w_vec = _itrf_uvw_frame(mjd, ra, dec)

        station_uv = np.zeros_like(station_ECEF)
        for iapos, apos in enumerate(station_ECEF):
            station_uv[iapos] = [
                np.dot(apos, u_vec),
                np.dot(apos, v_vec),
                np.dot(apos, w_vec),
            ]
        for bl in range(nbl):
            blants = antindices[bl]
            bla1 = blants[0]
            bla2 = blants[1]
            padded_uvw[ti * nbl + bl, :] = station_uv[bla1] - station_uv[bla2]

    return dict(
        zip(
            ["UVW", "TIME_CENTROID", "ANTENNA1", "ANTENNA2"],
            [padded_uvw, padded_time, padded_a1, padded_a2],
        )
    )


# Golden UVW values captured from the old python-casacore (pyrap) implementation
# (3 antennas, 2 timestamps, phase ref 1.53950448/-0.80037386, Signal Hill).
# The new erfa/astropy frame agrees with pyrap's casacore chain to <~1.7 arcsec
# of frame rotation, i.e. to well under 1e-4 m on these ~1 m baselines.
_PYRAP_GOLDEN_UVW = [
    [0.0, 0.0, 0.0],
    [0.56190225, -0.610954835, -0.594558705],
    [-0.805362649, -0.45141132, -0.436026175],
    [0.0, 0.0, 0.0],
    [-1.367264899, 0.159543515, 0.15853253],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.524077129, -0.62805505, -0.611056479],
    [-0.832042513, -0.425740258, -0.411035871],
    [0.0, 0.0, 0.0],
    [-1.356119642, 0.202314792, 0.200020608],
    [0.0, 0.0, 0.0],
]
_GOLDEN_STATION = np.array(
    [
        [-4486929.5, 584063.0, -4491740.0],
        [-4486930.25, 584063.692820323, -4491740.0],
        [-4486930.25, 584062.307179677, -4491740.0],
    ]
)
_GOLDEN_TIMES = np.array([5071671511.991, 5071672111.991])


class TestSynthesizeUVW(unittest.TestCase):
    """Test that the vectorized synthesize_uvw matches the original."""

    def _make_test_data(self, nant=24, ntime=10):
        """Create realistic test data mimicking TART telescope."""
        # Simulate a small antenna array: positions in ITRF (meters)
        # TART has antennas ~0.4m apart in a 2D plane, we scale up slightly
        np.random.seed(42)
        # Reference position (Signal Hill)
        base = np.array([-4486930.0, 584063.0, -4491740.0])
        # Spread antennas in a ~1m radius circle in the local tangent plane
        angles = np.linspace(0, 2 * np.pi, nant, endpoint=False)
        radius = 0.5
        local_east = radius * np.cos(angles)
        local_north = radius * np.sin(angles)
        local_up = np.zeros(nant)
        # Rough conversion to ITRF (approximate, but sufficient for testing
        # that both code paths produce identical results)
        station_ECEF = np.tile(base, (nant, 1)).astype(np.float64)
        station_ECEF[:, 0] += local_east * 0.5
        station_ECEF[:, 1] += local_north * 0.8
        station_ECEF[:, 2] += local_up

        # Generate timestamps spread over 10 minutes
        base_time = 5071671511.991  # ~2019-08-04 epoch
        times = np.linspace(base_time, base_time + 600, ntime)
        nbl = nant * (nant - 1) // 2 + nant

        # Build per-timestamp a1/a2 pairs (similar to what ms_create does)
        timems = np.repeat(times, nant)  # each timestamp has nant rows
        all_a1 = np.tile(np.arange(nant), ntime)
        all_a2 = np.tile(np.arange(nant), ntime)

        # Phase reference: zenith at Signal Hill at base_time
        # This is a rough RA/Dec but we're testing that both paths match,
        # not absolute correctness
        phase_ref = np.array([[1.53950448, -0.80037386]])  # radians (RA, Dec)

        return station_ECEF, timems, all_a1, all_a2, phase_ref

    def test_matches_original(self):
        """Core test: vectorized output must exactly match per-antenna loop."""
        station_ECEF, timems, a1, a2, phase_ref = self._make_test_data(
            nant=8, ntime=5
        )

        result_opt = synthesize_uvw(
            station_ECEF=station_ECEF,
            time=timems,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
            ack=False,
        )
        result_orig = _synthesize_uvw_original(
            station_ECEF=station_ECEF,
            time=timems,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
        )

        # Compare UVW arrays
        uvw_opt = result_opt["UVW"]
        uvw_orig = result_orig["UVW"]

        max_diff = np.max(np.abs(uvw_opt - uvw_orig))
        # Vectorized numpy (BLAS/FMA summation order) and the per-antenna
        # loop differ at float64 round-off on the ~1e6 m absolute station
        # coordinates (~2e-9 m here), not at bit level: 1e-8 keeps the
        # comparison far below any physically meaningful UVW error.
        self.assertLess(max_diff, 1e-8,
                        f"UVW mismatch: max diff = {max_diff}")

        # Compare metadata
        np.testing.assert_array_equal(result_opt["TIME_CENTROID"],
                                       result_orig["TIME_CENTROID"])
        np.testing.assert_array_equal(result_opt["ANTENNA1"],
                                       result_orig["ANTENNA1"])
        np.testing.assert_array_equal(result_opt["ANTENNA2"],
                                       result_orig["ANTENNA2"])

    def test_matches_pyrap_golden(self):
        """The new frame must reproduce the old python-casacore output.

        Golden values were captured from pyrap/casacore's ``to_uvw()`` before
        python-casacore was removed from the dependencies. The erfa/astropy
        frame agrees with that chain to <~1.7 arcsec of frame rotation, i.e.
        to ~1e-5 m on these ~1 m baselines (see fixvis._itrf_uvw_frame).
        """
        nant = 3
        ntime = 2
        timems = np.repeat(_GOLDEN_TIMES, nant)
        a1 = np.tile(np.arange(nant), ntime)
        a2 = np.tile(np.arange(nant), ntime)

        result = synthesize_uvw(
            station_ECEF=_GOLDEN_STATION,
            time=timems,
            a1=a1,
            a2=a2,
            phase_ref=np.array([[1.53950448, -0.80037386]]),
            ack=False,
        )
        max_diff = np.max(
            np.abs(result["UVW"] - np.array(_PYRAP_GOLDEN_UVW))
        )
        self.assertLess(
            max_diff, 1e-4,
            f"UVW drifted from the python-casacore reference: {max_diff}",
        )

    def test_large_array(self):
        """Test with realistic array size (24 antennas, 50 timestamps)."""
        station_ECEF, timems, a1, a2, phase_ref = self._make_test_data(
            nant=24, ntime=50
        )

        result_opt = synthesize_uvw(
            station_ECEF=station_ECEF,
            time=timems,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
            ack=False,
        )
        result_orig = _synthesize_uvw_original(
            station_ECEF=station_ECEF,
            time=timems,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
        )

        uvw_opt = result_opt["UVW"]
        uvw_orig = result_orig["UVW"]

        max_diff = np.max(np.abs(uvw_opt - uvw_orig))
        self.assertLess(max_diff, 1e-8,
                        f"UVW mismatch for 24-antenna array: max diff = {max_diff}")

    def test_single_timestamp(self):
        """Test with a single timestamp (edge case)."""
        station_ECEF, timems, a1, a2, phase_ref = self._make_test_data(
            nant=4, ntime=1
        )

        result_opt = synthesize_uvw(
            station_ECEF=station_ECEF,
            time=timems,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
            ack=False,
        )
        result_orig = _synthesize_uvw_original(
            station_ECEF=station_ECEF,
            time=timems,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
        )

        max_diff = np.max(np.abs(result_opt["UVW"] - result_orig["UVW"]))
        self.assertLess(max_diff, 1e-8)

    def test_different_phase_ref(self):
        """Test with a different phase reference direction."""
        station_ECEF, timems, a1, a2, _ = self._make_test_data(nant=6, ntime=3)
        # Point towards a different direction (DEC = 30 deg)
        phase_ref = np.array([[0.5, 0.52359878]])  # RA=28.6 deg, DEC=30 deg

        result_opt = synthesize_uvw(
            station_ECEF=station_ECEF,
            time=timems,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
            ack=False,
        )
        result_orig = _synthesize_uvw_original(
            station_ECEF=station_ECEF,
            time=timems,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
        )

        max_diff = np.max(np.abs(result_opt["UVW"] - result_orig["UVW"]))
        self.assertLess(max_diff, 1e-8)

    def test_repeated_antenna_indices(self):
        """Test with non-consecutive antenna indices (sparse configuration)."""
        station_ECEF, timems, _, _, phase_ref = self._make_test_data(
            nant=10, ntime=3
        )
        # Only use a subset of antennas (e.g., antennas 0,2,3,5,7,9)
        mask = np.isin(timems, np.unique(timems)[:])  # all rows
        a1 = np.array([0, 0, 0, 2, 2, 3, 5, 5, 7] * 3)
        a2 = np.array([2, 3, 5, 3, 7, 9, 7, 9, 9] * 3)
        timems_sparse = np.repeat(np.unique(timems), 9)

        result_opt = synthesize_uvw(
            station_ECEF=station_ECEF,
            time=timems_sparse,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
            ack=False,
        )
        result_orig = _synthesize_uvw_original(
            station_ECEF=station_ECEF,
            time=timems_sparse,
            a1=a1,
            a2=a2,
            phase_ref=phase_ref,
        )

        max_diff = np.max(np.abs(result_opt["UVW"] - result_orig["UVW"]))
        self.assertLess(max_diff, 1e-8)


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
