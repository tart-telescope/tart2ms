#
# Copyright Tim Molteno 2026 tim@elec.ac.nz
#
# Tests for the utility functions (TART online archive queries, issue #44)
#

import unittest
from datetime import datetime, timezone

from tart2ms.util import archive_query_window, parse_archive_query

NOW = datetime(2026, 1, 20, 12, 0, 0, tzinfo=timezone.utc)


class TestParseArchiveQuery(unittest.TestCase):

    def test_offset_query(self):
        name, start, interval, end = parse_archive_query("signal:-10:1:0")
        self.assertEqual(name, "signal")
        self.assertEqual(start, "-10")
        self.assertEqual(interval, 1.0)
        self.assertEqual(end, "0")

    def test_offset_query_with_interval(self):
        name, start, interval, end = parse_archive_query("rhodes:-100:10:0")
        self.assertEqual(name, "rhodes")
        self.assertEqual(start, "-100")
        self.assertEqual(interval, 10.0)
        self.assertEqual(end, "0")

    def test_timestamp_query(self):
        name, start, interval, end = parse_archive_query(
            "signal:2022-08-17T15:14:58:5:2022-08-17T16:14:58"
        )
        self.assertEqual(name, "signal")
        self.assertEqual(start, "2022-08-17T15:14:58")
        self.assertEqual(interval, 5.0)
        self.assertEqual(end, "2022-08-17T16:14:58")

    def test_timestamp_query_with_utc_offsets(self):
        # timestamps may contain colons and timezone offsets
        name, start, interval, end = parse_archive_query(
            "stellenbosch:2022-08-17T15:14:58+00:00:30:2022-08-17T16:14:58+00:00"
        )
        self.assertEqual(name, "stellenbosch")
        self.assertEqual(start, "2022-08-17T15:14:58+00:00")
        self.assertEqual(interval, 30.0)
        self.assertEqual(end, "2022-08-17T16:14:58+00:00")

    def test_timestamp_query_with_z_suffix(self):
        name, start, interval, end = parse_archive_query(
            "signal:2022-08-17T15:14:58Z:1:2022-08-17T15:20:00Z"
        )
        self.assertEqual(start, "2022-08-17T15:14:58Z")
        self.assertEqual(interval, 1.0)
        self.assertEqual(end, "2022-08-17T15:20:00Z")

    def test_mixed_offset_and_timestamp(self):
        name, start, interval, end = parse_archive_query(
            "signal:-10:1:2026-01-20T12:00:00"
        )
        self.assertEqual(start, "-10")
        self.assertEqual(end, "2026-01-20T12:00:00")

    def test_bad_queries_raise(self):
        for bad in [
            "",
            "signal",
            "signal:-10:1",  # missing END
            "signal::1:0",  # missing START
            "signal:-10::0",  # missing INTERVAL
            "signal:-10:once:0",  # non-numeric INTERVAL
            "signal:-10:-1:0",  # non-positive interval
            "signal:-10:1:0:0",  # too many fields
        ]:
            with self.assertRaises(ValueError):
                parse_archive_query(bad)


class TestArchiveQueryWindow(unittest.TestCase):

    def test_offset_window(self):
        start, end = archive_query_window("-10", "0", now=NOW)
        self.assertEqual(end, NOW)
        self.assertEqual((end - start).total_seconds(), 600.0)

    def test_timestamp_window(self):
        start, end = archive_query_window(
            "2022-08-17T15:14:58", "2022-08-17T16:14:58", now=NOW
        )
        self.assertEqual(start, datetime(2022, 8, 17, 15, 14, 58, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2022, 8, 17, 16, 14, 58, tzinfo=timezone.utc))
        self.assertEqual((end - start).total_seconds(), 3600.0)

    def test_timestamps_are_utc(self):
        start, end = archive_query_window(
            "2022-08-17T17:14:58+02:00", "2022-08-17T16:14:58+00:00", now=NOW
        )
        self.assertEqual(start, datetime(2022, 8, 17, 15, 14, 58, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2022, 8, 17, 16, 14, 58, tzinfo=timezone.utc))

    def test_end_before_start_raises(self):
        with self.assertRaises(ValueError):
            archive_query_window("0", "-10", now=NOW)
        with self.assertRaises(ValueError):
            archive_query_window(
                "2022-08-17T16:14:58", "2022-08-17T15:14:58", now=NOW
            )


if __name__ == "__main__":
    unittest.main()
