#
# Unit test for predict_model optimizations.
# Verifies that hoisting .compute() and np.array() out of the
# per-timestamp loop produces identical output.
#

import unittest
import logging
import json

import numpy as np
import dask.array as da
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
from tart.operation import settings
from tart_tools import api_imaging

from tart2ms.ms_helper import predict_model

logger = logging.getLogger("tart2ms")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

AFRICANUS_DFT_AVAIL = True
try:
    from africanus.rime.dask import wsclean_predict
    from africanus.coordinates.dask import radec_to_lm
except ImportError:
    AFRICANUS_DFT_AVAIL = False


# Reference: original implementation before optimizations,
# with .compute() and np.array() inside the loop on every iteration.
def _predict_model_original(
    dask_data_shape,
    dask_data_chunking,
    dask_data_dtype,
    uvw_data,
    epoch_s,
    spw_chan_freqs,
    spw_i,
    zenith_directions,
    map_row_to_zendir,
    location,
    sources,
    epoch_s_sources,
    sources_obstime,
    writemodelcatalog,
    filter_elevation=45.0,
    filter_name=None,
    cat_name_prefix="model_sources_",
    append_catalog=False,
    default_flux=1e5,
):
    """Original per-timestamp loop without pre-computation."""
    import re
    from tart2ms.ms_helper import azel2radec

    if filter_name is None:
        filter_name = (
            r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)"
            r"|(?:^CELESTIAL_.*)|(?:^SOLAR_.*)"
        )

    if not AFRICANUS_DFT_AVAIL:
        return None
    if not sources:
        return None

    if len(epoch_s_sources) != len(sources):
        raise RuntimeError("epoch_s_sources and sources must have same length")

    model_data = da.zeros(dask_data_shape, chunks=dask_data_chunking,
                          dtype=dask_data_dtype)
    spwi_chan_freqs = da.from_array(spw_chan_freqs[spw_i])

    for dataset_i, data_epoch_i in enumerate(epoch_s):
        nn_source_epoch = np.argmin(abs(np.array(epoch_s_sources) - data_epoch_i))
        epoch_s_i = epoch_s_sources[nn_source_epoch]

        if sources[nn_source_epoch] is None:
            continue

        sources_i = list(filter(
            lambda s: s.get('el', -90) >= filter_elevation
                      and re.findall(filter_name, s.get('name', 'NULLPTR')),
            sources[nn_source_epoch]
        ))
        if not sources_i:
            continue

        sources_radec = np.empty((len(sources_i), 2))
        names = []
        for src_i, src in enumerate(sources_i):
            names.append(src['name'].replace(" ", "_")
                         .replace("CELESTIAL_", "")
                         .replace("SOLAR_", ""))
            direction_src = azel2radec(
                az=src['az'], el=src['el'],
                distance=src.get('distance', None),
                location=location,
                obstime=sources_obstime[nn_source_epoch],
            )
            sources_radec[src_i, :] = direction_src

        zenith_i = zenith_directions[dataset_i]
        lm = radec_to_lm(sources_radec, zenith_i)
        source_type = np.array(["POINT"] * len(sources_i))
        gauss_shape = np.stack(
            ([0.0] * len(sources_i),
             [0.0] * len(sources_i),
             [0.0] * len(sources_i)),
            axis=-1,
        )
        flux = np.ones(len(sources_i))
        for ssi, ss in enumerate(sources_i):
            flux[ssi] = ss.get("flux", lambda nu: default_flux)(
                np.mean(spw_chan_freqs[spw_i])
            )

        spi = np.zeros((len(sources_i), 1))
        reffreq = np.ones(len(sources_i)) * np.mean(spw_chan_freqs[spw_i])
        logspi = np.ones(len(sources_i), dtype=bool)

        sel = map_row_to_zendir.compute() == dataset_i
        vis = wsclean_predict(
            uvw_data[sel, :], lm, source_type, flux,
            spi, logspi, reffreq, gauss_shape, spwi_chan_freqs,
        )
        model_data[sel, :, :] = vis

    return model_data


class TestPredictModel(unittest.TestCase):
    """Test that optimized predict_model matches original output."""

    @classmethod
    def setUpClass(cls):
        if not AFRICANUS_DFT_AVAIL:
            raise unittest.SkipTest("codex-africanus not available")

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

        cls.timestamps = [timestamp]
        cls.obstime = Time(cls.timestamps)
        cls.epoch_s = [ts.timestamp() for ts in cls.timestamps]
        cls.source_json = d[1]
        cls.spw_chan_freqs = [np.array([info["info"]["operating_frequency"]])]
        cls._zenith_dir = np.array([1.5395, -0.8004])

    def _run_both(self, nant=8, ntime=3):
        """Run both implementations and compare."""
        nbl = nant * (nant - 1) // 2 + nant
        nrows = nbl * ntime

        shape = (nrows, 1, 1)
        chunks = (nbl, 1, 1)

        np.random.seed(42)
        uvw_data = da.from_array(
            np.random.randn(nrows, 3).astype(np.float64),
            chunks=(nbl, 3),
        )

        # Each timestamp has nbl rows, all mapping to same zenith index
        zendir_arr = np.repeat(np.arange(ntime), nbl)
        map_row_to_zendir = da.from_array(zendir_arr, chunks=nbl)

        # zenith_directions must have one row per unique timestamp
        zenith_directions = np.tile(self._zenith_dir, (ntime, 1))

        epoch_s = np.linspace(
            self.epoch_s[0], self.epoch_s[0] + 600, ntime
        ).tolist()
        epoch_s_src = [self.epoch_s[0]] * ntime  # reuse same catalog
        sources = [self.source_json] * ntime

        # Run optimized (current)
        result_opt = predict_model(
            shape, chunks, np.complex128,
            uvw_data, epoch_s, self.spw_chan_freqs, 0,
            zenith_directions, map_row_to_zendir, self.location,
            sources, epoch_s_src, self.obstime,
            writemodelcatalog=False,
        )

        # Run original
        result_orig = _predict_model_original(
            shape, chunks, np.complex128,
            uvw_data, epoch_s, self.spw_chan_freqs, 0,
            zenith_directions, map_row_to_zendir, self.location,
            sources, epoch_s_src, self.obstime,
            writemodelcatalog=False,
        )

        return result_opt, result_orig

    def test_matches_original(self):
        """Optimized output must match original bit-for-bit."""
        opt, orig = self._run_both(nant=4, ntime=2)

        opt_np = opt.compute()
        orig_np = orig.compute()

        max_diff = np.max(np.abs(opt_np - orig_np))
        self.assertLess(
            max_diff, 1e-12,
            f"predict_model output mismatch: max diff = {max_diff}"
        )

    def test_single_timestamp(self):
        """Edge case: single timestamp."""
        opt, orig = self._run_both(nant=4, ntime=1)
        opt_np = opt.compute()
        orig_np = orig.compute()
        max_diff = np.max(np.abs(opt_np - orig_np))
        self.assertLess(max_diff, 1e-12)

    def test_multiple_timestamps_same_catalog(self):
        """Multiple timestamps all pointing to same source catalog epoch."""
        opt, orig = self._run_both(nant=6, ntime=5)
        opt_np = opt.compute()
        orig_np = orig.compute()
        max_diff = np.max(np.abs(opt_np - orig_np))
        self.assertLess(max_diff, 1e-12)


if __name__ == "__main__":
    unittest.main()
