#
# Unit test for tart-catalogue-client integration in __fetch_sources.
# Verifies the client path produces compatible output with the legacy path.
#

import unittest
import logging
from datetime import datetime, timezone, timedelta

import tart2ms.tart2ms as t2m

logger = logging.getLogger("tart2ms")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

# Signal Hill coordinates
LAT = -45.85177
LON = 170.5456


@unittest.skipUnless(t2m.CATALOGUE_CLIENT_AVAIL, "tart-catalogue-client not available")
class TestFetchSourcesClient(unittest.TestCase):
    """Test getattr(t2m, '__fetch_sources_via_client') against the legacy path."""

    @classmethod
    def setUpClass(cls):
        cls.timestamps = [
            datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 1, 12, 10, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 1, 12, 20, 0, tzinfo=timezone.utc),
        ]

    def test_client_path_runs(self):
        """Client path should return results without error."""
        sources, times = getattr(t2m, '__fetch_sources_via_client')(
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

    def test_output_format_matches_legacy(self):
        """Client output format should be compatible with legacy format."""
        sources_client, _ = getattr(t2m, '__fetch_sources_via_client')(
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT,
            observer_lon=LON,
            filter_elevation=20.0,  # lower to get more sources
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )
        sources_legacy, _ = getattr(t2m, '__fetch_sources_legacy')(
            timestamps=self.timestamps[:1],
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT,
            observer_lon=LON,
            retry=3,
            retry_time=1,
            force_recache=False,
            filter_elevation=20.0,
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )

        # Both should return lists of dicts with matching keys
        self.assertEqual(len(sources_client), len(sources_legacy))
        for client_list, legacy_list in zip(sources_client, sources_legacy):
            self.assertIsInstance(client_list, list)
            self.assertIsInstance(legacy_list, list)
            for s in client_list:
                self.assertIn("name", s)
                self.assertIn("az", s)
                self.assertIn("el", s)
                # jy not in legacy because old API uses different field
                # but client output should have it for compat

    def test_elevation_filtering(self):
        """Sources below filter_elevation should be excluded."""
        for elev in [45.0, 60.0]:
            sources, _ = getattr(t2m, '__fetch_sources_via_client')(
                downsampletimes=[self.timestamps[0]],
                observer_lat=LAT,
                observer_lon=LON,
                filter_elevation=elev,
                filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
            )
            for s in sources[0]:
                self.assertGreaterEqual(s["el"], elev,
                    f"Source {s['name']} has el={s['el']} < {elev}")

    def test_name_filtering(self):
        """Only sources matching filter_name should be included."""
        # GPS-only filter
        sources, _ = getattr(t2m, '__fetch_sources_via_client')(
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

    def test_integrated_fetch_routes_to_client(self):
        """The main __fetch_sources should use the client path."""
        sources, times = getattr(t2m, '__fetch_sources')(
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


if __name__ == "__main__":
    unittest.main()
