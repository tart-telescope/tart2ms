#
# Unit test for the vectorized synthesize_uvw optimization.
# Compares output of the optimized function against the original
# per-antenna loop implementation to ensure correctness.
#

import unittest
import logging

import numpy as np
from pyrap.measures import measures
from pyrap.quanta import quantity

from tart2ms.fixvis import synthesize_uvw

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
    """Original per-antenna loop implementation for comparison testing."""
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

    dm = measures()
    epoch = dm.epoch(time_TZ, quantity(time[0], time_unit))
    refdir = dm.direction(
        stopctr_epoch,
        quantity(phase_ref[0, 0], stopctr_units[0]),
        quantity(phase_ref[0, 1], stopctr_units[1]),
    )
    obs = dm.position(
        posframe,
        quantity(station_ECEF[0, 0], posunits[0]),
        quantity(station_ECEF[0, 1], posunits[1]),
        quantity(station_ECEF[0, 2], posunits[2]),
    )

    dm.do_frame(obs)
    dm.do_frame(refdir)
    dm.do_frame(epoch)
    for ti, t in enumerate(unique_time):
        epoch = dm.epoch(time_TZ, quantity(t, "s"))
        dm.do_frame(epoch)

        station_uv = np.zeros_like(station_ECEF)
        for iapos, apos in enumerate(station_ECEF):
            compuvw = dm.to_uvw(
                dm.baseline(
                    posframe,
                    quantity([apos[0], station_ECEF[0, 0]], posunits[0]),
                    quantity([apos[1], station_ECEF[0, 1]], posunits[1]),
                    quantity([apos[2], station_ECEF[0, 2]], posunits[2]),
                )
            )
            station_uv[iapos] = compuvw["xyz"].get_value()[0:3]
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
        self.assertLess(max_diff, 1e-12,
                        f"UVW mismatch: max diff = {max_diff}")

        # Compare metadata
        np.testing.assert_array_equal(result_opt["TIME_CENTROID"],
                                       result_orig["TIME_CENTROID"])
        np.testing.assert_array_equal(result_opt["ANTENNA1"],
                                       result_orig["ANTENNA1"])
        np.testing.assert_array_equal(result_opt["ANTENNA2"],
                                       result_orig["ANTENNA2"])

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
        self.assertLess(max_diff, 1e-12,
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
        self.assertLess(max_diff, 1e-12)

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
        self.assertLess(max_diff, 1e-12)

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
        self.assertLess(max_diff, 1e-12)


if __name__ == "__main__":
    unittest.main()
