import unittest
from datetime import timedelta

from tinyCode.time_utils import (
    BEIJING_TIMEZONE,
    beijing_filename_timestamp,
    beijing_now,
    beijing_now_iso,
    format_beijing_time,
    timestamp_sort_key,
    to_beijing,
)


class TimeUtilsTests(unittest.TestCase):
    def test_beijing_now_is_timezone_aware(self):
        now = beijing_now()

        self.assertEqual(timedelta(hours=8), now.utcoffset())
        self.assertIs(BEIJING_TIMEZONE, now.tzinfo)

    def test_iso_timestamp_contains_explicit_beijing_offset(self):
        timestamp = beijing_now_iso()

        self.assertTrue(timestamp.endswith("+08:00"), timestamp)

    def test_filename_timestamp_is_sortable_and_safe(self):
        stamp = beijing_filename_timestamp(microseconds=True)

        self.assertRegex(stamp, r"^\d{8}_\d{6}_\d{6}$")

    def test_legacy_utc_timestamp_is_displayed_as_beijing_time(self):
        displayed = format_beijing_time("2026-09-01T02:30:00+00:00")

        self.assertEqual("2026-09-01 10:30:00 北京时间", displayed)

    def test_legacy_naive_timestamp_is_treated_as_utc(self):
        parsed = to_beijing("2026-09-01T02:30:00")

        self.assertIsNotNone(parsed)
        self.assertEqual(10, parsed.hour)
        self.assertEqual(timedelta(hours=8), parsed.utcoffset())

    def test_sort_key_compares_mixed_offsets_by_actual_instant(self):
        earlier = timestamp_sort_key("2026-09-01T09:00:00+08:00")
        later = timestamp_sort_key("2026-09-01T02:00:00+00:00")

        self.assertLess(earlier, later)


if __name__ == "__main__":
    unittest.main()
