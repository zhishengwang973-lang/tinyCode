import json
import os
import tempfile
import unittest
from pathlib import Path

from tinyCode.teams.lease import TeamRunLease


class TeamRunLeaseTests(unittest.TestCase):
    def test_second_live_process_lease_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = TeamRunLease(root)
            second = TeamRunLease(root)
            self.assertTrue(first.acquire()[0])

            ok, message = second.acquire()

            self.assertFalse(ok)
            self.assertIn("不能并发启动", message)
            first.release()

    def test_stale_lease_is_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "run.lock"
            lock.write_text(
                json.dumps({"pid": 999_999_999, "token": "stale"}),
                encoding="utf-8",
            )
            lease = TeamRunLease(root)

            ok, message = lease.acquire()

            self.assertTrue(ok, message)
            owner = json.loads(lock.read_text(encoding="utf-8"))
            self.assertEqual(os.getpid(), owner["pid"])
            lease.release()


if __name__ == "__main__":
    unittest.main()
