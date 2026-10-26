#
# Unit test for tart-catalogue-client integration in __fetch_sources.
# Verifies celestial_positions produces correct ra/dec in radians.
#

import unittest
import logging
from datetime import datetime, timezone

import numpy as np

import tart2ms.tart2ms as t2m

logger = logging.getLogger("tart2ms")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

LAT = -45.85177
LON = 170.5456

_fetch = getattr(t2m, "__fetch_sources")
_fetch_via = getattr(t2m, "__fetch_sources_via_client")


class TestFetchSourcesClient(unittest.TestCase):
    """Test __fetch_sources via tart-catalogue-client celestial_positions."""

    @classmethod
    def setUpClass(cls):
        cls.timestamps = [
            datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 1, 12, 10, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 1, 12, 20, 0, tzinfo=timezone.utc),
        ]

    def test_client_path_runs(self):
        sources, times = _fetch_via(
            downsampletimes=self.timestamps,
            observer_lat=LAT,
            observer_lon=LON,
            filter_elevation=-90.0,  # accept all for this test
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )
        self.assertEqual(len(sources), len(self.timestamps))
        for src in sources:
            self.assertIsInstance(src, list)
            for s in src:
                self.assertIn("name", s)
                self.assertIn("ra", s)
                self.assertIn("dec", s)
                self.assertIn("az", s)
                self.assertIn("el", s)
                self.assertIn("jy", s)
                self.assertIsInstance(s["ra"], (int, float, np.floating))
                self.assertIsInstance(s["dec"], (int, float, np.floating))
                self.assertIsInstance(s["az"], (int, float, np.floating))
                self.assertIsInstance(s["el"], (int, float, np.floating))

    def test_ra_dec_in_radians(self):
        sources, _ = _fetch_via(
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT,
            observer_lon=LON,
            filter_elevation=-90.0,
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )
        for s in sources[0]:
            # RA should be in radians (0 to 2π)
            self.assertGreaterEqual(s["ra"], 0.0)
            self.assertLessEqual(s["ra"], 2 * np.pi)
            # DEC should be in radians (-π/2 to π/2)
            self.assertGreaterEqual(s["dec"], -np.pi / 2)
            self.assertLessEqual(s["dec"], np.pi / 2)

    def test_name_filtering(self):
        sources, _ = _fetch_via(
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT,
            observer_lon=LON,
            filter_elevation=-90.0,
            filter_name=r"^GPS.*",
        )
        for s in sources[0]:
            self.assertTrue(
                s["name"].startswith("GPS"),
                f"Source {s['name']} doesn't match GPS filter"
            )

    def test_integrated_fetch(self):
        sources, times = _fetch(
            timestamps=self.timestamps,
            observer_lat=LAT,
            observer_lon=LON,
            filter_elevation=-90.0,
            downsample=1.0,
        )
        self.assertEqual(len(sources), len(self.timestamps))
        self.assertEqual(len(times), len(self.timestamps))
        for src in sources:
            for s in src:
                self.assertIn("ra", s)
                self.assertIn("dec", s)
                self.assertIn("jy", s)

    def test_elevation_filtering(self):
        """Sources below the topocentric elevation threshold should be excluded.

        Note: filter_elevation is a true horizon elevation cut (deg), as used
        by predict_model's `el` filter -- not a declination cut. (This test
        previously asserted a declination threshold, which combined with the
        missing 'el' key silently dropped all client-fetched GNSS sources
        from the model and the SOURCE table.)
        """
        for elev in [-45.0, -30.0, 0.0]:
            sources, _ = _fetch_via(
                downsampletimes=[self.timestamps[0]],
                observer_lat=LAT,
                observer_lon=LON,
                filter_elevation=elev,
                filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
            )
            self.assertGreater(len(sources[0]), 0,
                               f"expected some GNSS sources above {elev} deg el")
            for s in sources[0]:
                self.assertGreaterEqual(
                    s["el"], elev,
                    f"Source {s['name']} has el={s['el']:.1f} < {elev}"
                )
        # a near-zenith cut must shrink the source list
        sources_low, _ = _fetch_via(
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT, observer_lon=LON,
            filter_elevation=0.0,
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )
        sources_high, _ = _fetch_via(
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT, observer_lon=LON,
            filter_elevation=60.0,
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )
        self.assertLessEqual(len(sources_high[0]), len(sources_low[0]))
        for s in sources_high[0]:
            self.assertGreaterEqual(s["el"], 60.0)


if __name__ == "__main__":
    unittest.main()
