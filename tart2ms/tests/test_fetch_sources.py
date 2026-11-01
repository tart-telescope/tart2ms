#
# Unit test for tart-catalogue-client integration in __fetch_sources.
# Verifies sources carry topocentric az/el (and no barycentric ra/dec --
# see issue #68: celestial_positions() ra/dec pointed every GNSS source
# at the Earth's barycentric position).
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
    """Test __fetch_sources via tart-catalogue-client (topocentric az/el)."""

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
                self.assertIn("az", s)
                self.assertIn("el", s)
                self.assertIn("jy", s)
                self.assertIsInstance(s["az"], (int, float, np.floating))
                self.assertIsInstance(s["el"], (int, float, np.floating))
                self.assertIsInstance(s["jy"], (int, float, np.floating))

    def test_no_barycentric_ra_dec(self):
        """Client-fetched GNSS sources must not carry 'ra'/'dec' (issue #68).

        celestial_positions() returns barycentric ICRS directions that point
        every GNSS satellite at the Earth's barycentric position
        (tart-telescope/catalogue issue #14). Sources must therefore carry
        topocentric az/el only, so predict_model and the SOURCE table writer
        convert via azel2radec (the verified-correct path).
        """
        sources, _ = _fetch_via(
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT,
            observer_lon=LON,
            filter_elevation=-90.0,
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )
        self.assertGreater(len(sources[0]), 0)
        for s in sources[0]:
            self.assertNotIn(
                "ra", s,
                f"Source {s['name']} carries barycentric 'ra' (issue #68)")
            self.assertNotIn(
                "dec", s,
                f"Source {s['name']} carries barycentric 'dec' (issue #68)")
            # topocentric az/el are well-defined degrees
            self.assertGreaterEqual(s["az"], 0.0)
            self.assertLessEqual(s["az"], 360.0)
            self.assertGreaterEqual(s["el"], -90.0)
            self.assertLessEqual(s["el"], 90.0)

    def test_model_directions_spread_over_sky(self):
        """azel2radec on the fetched az/el must spread GNSS sources over the sky.

        This is the observable symptom of issue #68: with the barycentric
        ra/dec pass-through, every GNSS source predicted at a single sky
        position (max pairwise separation ~0). The topocentric az/el path
        must instead yield widely separated directions.
        """
        from astropy.coordinates import EarthLocation, SkyCoord
        import astropy.units as u
        from tart2ms.ms_helper import azel2radec

        sources, _ = _fetch_via(
            downsampletimes=[self.timestamps[0]],
            observer_lat=LAT,
            observer_lon=LON,
            filter_elevation=-90.0,
            filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^BEIDOU.*)|(?:^GSAT.*)",
        )
        srcs = sources[0]
        self.assertGreater(len(srcs), 1)
        location = EarthLocation.from_geodetic(
            lon=LON * u.deg, lat=LAT * u.deg
        )
        az = np.array([s["az"] for s in srcs])
        el = np.array([s["el"] for s in srcs])
        ra, dec = azel2radec(
            az=az, el=el, location=location,
            obstime=self.timestamps[0],
        )
        coords = SkyCoord(ra=np.asarray(ra) * u.rad,
                          dec=np.asarray(dec) * u.rad)
        max_sep = max(
            coords[i].separation(coords[j]).deg
            for i in range(len(coords) - 1)
            for j in range(i + 1, len(coords))
        )
        self.assertGreater(
            max_sep, 10.0,
            f"GNSS model directions cluster at one position "
            f"(max separation {max_sep:.3f} deg) -- issue #68"
        )

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
                # no barycentric ra/dec pass-through (issue #68)
                self.assertNotIn("ra", s)
                self.assertNotIn("dec", s)
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
