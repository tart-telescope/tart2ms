#
# Unit test for tart-catalogue-client integration in __fetch_sources.
#

import unittest
import logging
from datetime import datetime, timezone

import tart2ms.tart2ms as t2m

logger = logging.getLogger("tart2ms")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

# Signal Hill coordinates
LAT = -45.85177
LON = 170.5456

_fetch = getattr(t2m, "__fetch_sources")
_fetch_via = getattr(t2m, "__fetch_sources_via_client")


class TestFetchSourcesClient(unittest.TestCase):
    """Test __fetch_sources via tart-catalogue-client."""

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
            filter_elevation=45.0,
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )
        self.assertEqual(len(sources), len(self.timestamps))
        for src in sources:
            self.assertIsInstance(src, list)
            for s in src:
                self.assertIn("name", s)
                self.assertIn("az", s)
                self.assertIn("el", s)
                self.assertIn("jy", s)
                self.assertGreaterEqual(s["el"], 45.0)

    def test_elevation_filtering(self):
        for elev in [45.0, 60.0]:
            sources, _ = _fetch_via(
                downsampletimes=[self.timestamps[0]],
                observer_lat=LAT,
                observer_lon=LON,
                filter_elevation=elev,
                filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
            )
            for s in sources[0]:
                self.assertGreaterEqual(
                    s["el"], elev,
                    f"Source {s['name']} has el={s['el']} < {elev}"
                )

    def test_name_filtering(self):
        sources, _ = _fetch_via(
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT,
            observer_lon=LON,
            filter_elevation=10.0,
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
            filter_elevation=45.0,
            downsample=1.0,
        )
        self.assertEqual(len(sources), len(self.timestamps))
        self.assertEqual(len(times), len(self.timestamps))
        for src in sources:
            for s in src:
                self.assertIn("az", s)
                self.assertIn("el", s)
                self.assertIn("jy", s)
                self.assertGreaterEqual(s["el"], 45.0)

    def test_output_format(self):
        sources, _ = _fetch_via(
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT,
            observer_lon=LON,
            filter_elevation=20.0,
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )
        for s in sources[0]:
            self.assertIn("name", s)
            self.assertIn("az", s)
            self.assertIn("el", s)
            self.assertIn("jy", s)
            self.assertIsInstance(s["az"], (int, float))
            self.assertIsInstance(s["el"], (int, float))
            self.assertIsInstance(s["jy"], (int, float))


if __name__ == "__main__":
    unittest.main()
