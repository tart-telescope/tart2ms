#
# Unit test for SOURCE table construction optimization.
# Verifies that building a single contiguous numpy array (wrapped in dask)
# produces identical output to the old per-source dask array approach.
#

import unittest
import logging
import json

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
from tart.operation import settings
from tart_tools import api_imaging

from tart2ms.ms_helper import azel2radec

logger = logging.getLogger("tart2ms")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)


def _build_source_dataset_original(sources, epoch_s_sources, sources_obstime, location):
    """Original implementation: per-source dask arrays + concatenation.

    Returns a dict of numpy arrays extracted from the dask Dataset,
    matching what would be written to the MS SOURCE table.
    """
    import dask.array as da

    if len(epoch_s_sources) != len(sources):
        raise RuntimeError("epoch_s_sources and sources mismatch")

    all_numlines = []
    all_name = []
    all_time = []
    all_direction = []

    for database_i, (epoch_s_i, sources_i) in enumerate(
        zip(epoch_s_sources, sources)
    ):
        if sources_i is None:
            continue
        for src in sources_i:
            name = src["name"]
            direction_src = azel2radec(
                az=src["az"],
                el=src["el"],
                location=location,
                obstime=sources_obstime[database_i],
            )
            dask_num_lines = da.asarray(np.asarray([1], dtype=np.int32))
            dask_direction = da.asarray(
                np.asarray(direction_src, dtype=np.float64), chunks=1
            )[None, :]
            dask_name = da.asarray(np.asarray([name], dtype=object), chunks=1)
            dask_time = da.asarray(np.asarray([epoch_s_i], dtype=object), chunks=1)
            all_numlines.append(dask_num_lines)
            all_name.append(dask_name)
            all_time.append(dask_time)
            all_direction.append(dask_direction)

    result = {
        "NUM_LINES": da.concatenate(all_numlines, axis=0).compute(),
        "NAME": da.concatenate(all_name, axis=0).compute(),
        "TIME": da.concatenate(all_time, axis=0).compute(),
        "DIRECTION": da.concatenate(all_direction, axis=0).compute(),
    }
    return result


def _build_source_dataset_optimized(sources, epoch_s_sources, sources_obstime, location):
    """Optimized: build numpy lists first, wrap in dask once.

    Returns a dict of numpy arrays matching what would be written
    to the MS SOURCE table.
    """
    import dask.array as da

    if len(epoch_s_sources) != len(sources):
        raise RuntimeError("epoch_s_sources and sources mismatch")

    all_numlines = []
    all_name = []
    all_time = []
    all_direction = []

    for database_i, (epoch_s_i, sources_i) in enumerate(
        zip(epoch_s_sources, sources)
    ):
        if sources_i is None:
            continue
        for src in sources_i:
            name = src["name"]
            direction_src = azel2radec(
                az=src["az"],
                el=src["el"],
                location=location,
                obstime=sources_obstime[database_i],
            )
            all_numlines.append(1)
            all_name.append(name)
            all_time.append(epoch_s_i)
            all_direction.append(direction_src)

    n_src = len(all_numlines)
    # Wrap as dask arrays (mimicking the main code) then compute to compare
    result = {
        "NUM_LINES": da.from_array(
            np.array(all_numlines, dtype=np.int32), chunks=n_src
        ).compute(),
        "NAME": da.from_array(
            np.array(all_name, dtype=object), chunks=n_src
        ).compute(),
        "TIME": da.from_array(
            np.array(all_time, dtype=object), chunks=n_src
        ).compute(),
        "DIRECTION": da.from_array(
            np.array(all_direction, dtype=np.float64), chunks=n_src
        ).compute(),
    }
    return result


class TestSourceTable(unittest.TestCase):
    """Test that optimized SOURCE table construction matches original."""

    @classmethod
    def setUpClass(cls):
        with open("tart2ms/tests/data_test.json") as f:
            jdi = json.load(f)
        info = jdi["info"]
        loc = info["info"]["location"]
        cls.location = EarthLocation.from_geodetic(
            lon=loc["lon"] * u.deg,
            lat=loc["lat"] * u.deg,
            height=loc["alt"] * u.m,
            ellipsoid="WGS84",
        )
        config = settings.from_api_json(info["info"], jdi["ant_pos"])
        d = jdi["data"][0]
        gains = np.array(jdi["gains"]["gain"])
        phases = np.array(jdi["gains"]["phase_offset"])
        cal_vis, timestamp = api_imaging.vis_calibrated(
            d[0], config, gains, phases, []
        )
        cls.timestamp = timestamp
        cls.obstime = Time([timestamp])
        cls.source_json = d[1]

    def _run_both(self, n_catalogs=1):
        """Run both implementations with n_catalogs copies of the test data."""
        epoch_s_sources = [self.timestamp.timestamp()] * n_catalogs
        sources = [self.source_json] * n_catalogs
        sources_obstime = Time([self.timestamp] * n_catalogs)

        result_orig = _build_source_dataset_original(
            sources, epoch_s_sources, sources_obstime, self.location
        )
        result_opt = _build_source_dataset_optimized(
            sources, epoch_s_sources, sources_obstime, self.location
        )
        return result_orig, result_opt

    def test_matches_original_single_catalog(self):
        """Single catalog: output must match exactly."""
        orig, opt = self._run_both(n_catalogs=1)

        np.testing.assert_array_equal(orig["NUM_LINES"], opt["NUM_LINES"])
        np.testing.assert_array_equal(orig["NAME"], opt["NAME"])
        np.testing.assert_array_equal(orig["TIME"], opt["TIME"])
        np.testing.assert_array_almost_equal(
            orig["DIRECTION"], opt["DIRECTION"], decimal=12
        )

    def test_matches_original_multiple_catalogs(self):
        """Multiple catalogs: output must match exactly."""
        orig, opt = self._run_both(n_catalogs=3)

        np.testing.assert_array_equal(orig["NUM_LINES"], opt["NUM_LINES"])
        np.testing.assert_array_equal(orig["NAME"], opt["NAME"])
        np.testing.assert_array_equal(orig["TIME"], opt["TIME"])
        np.testing.assert_array_almost_equal(
            orig["DIRECTION"], opt["DIRECTION"], decimal=12
        )

    def test_shapes_consistent(self):
        """All columns should have consistent row counts."""
        _, opt = self._run_both(n_catalogs=2)

        nrows = len(opt["NUM_LINES"])
        self.assertEqual(opt["NAME"].shape, (nrows,))
        self.assertEqual(opt["TIME"].shape, (nrows,))
        self.assertEqual(opt["DIRECTION"].shape, (nrows, 2))
        self.assertGreater(nrows, 0)

    def test_direction_two_columns(self):
        """DIRECTION must have shape (n, 2) — RA and DEC per source."""
        _, opt = self._run_both(n_catalogs=1)
        self.assertEqual(opt["DIRECTION"].ndim, 2)
        self.assertEqual(opt["DIRECTION"].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
