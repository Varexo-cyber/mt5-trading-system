import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.types import Direction
from scripts import replay_requested_markets as launch
from scripts.dry_run_sections import _resolve, build_parser, main


class RequestedReplayTests(unittest.TestCase):
    def test_be_exit_is_earlier_than_fixed_loss(self):
        index = pd.date_range("2026-06-01", periods=3, freq="min", tz="UTC")
        frame = pd.DataFrame({"high": [104, 103, 100], "low": [99, 100, 89],
                              "close": [103, 101, 90]}, index=index)
        idea = SimpleNamespace(direction=Direction.LONG, entry=100, stop_loss=90, take_profit=110)
        fixed, fixed_at, managed, managed_at = _resolve(
            frame, index[0], idea, 3, manage=(0.25, 1.0)
        )
        self.assertEqual(fixed, -1.0)
        self.assertAlmostEqual(managed, 0.1)
        self.assertEqual(managed_at, index[1])
        self.assertEqual(fixed_at, index[2])

    def test_us30_separate_clocks_and_original_detectors(self):
        runs = [build_parser().parse_args(c) for c in launch.commands("us30", 180)]
        self.assertEqual([r.sweep for r in runs], [["M1"], ["M5"]])
        for r in runs:
            self.assertEqual(r.only, "impulse_retest,order_block_fast")
            self.assertEqual(r.symbols, "US30")
            self.assertTrue(r.jarvis_replay)
            self.assertFalse(r.live_only)

    def test_s5_two_independent_runs(self):
        with patch.object(launch, "replay") as call:
            launch.main(["s5", "90", "--database", "archive.sqlite3", "--equity", "258.90"])
        self.assertEqual(call.call_count, 2)
        runs = [build_parser().parse_args(c.args[0]) for c in call.call_args_list]
        self.assertEqual([r.s5_exit for r in runs], ["fixed", "break-even"])
        self.assertNotEqual(runs[0].csv, runs[1].csv)
        for r in runs:
            self.assertEqual(r.database, "archive.sqlite3")
            self.assertEqual(r.only, "section_five_ndx100_m5")
            self.assertFalse(r.sweep)

    def test_override_refuses_other_sections_before_loading_or_connecting(self):
        with patch("scripts.dry_run_sections.load_settings") as load:
            with self.assertRaises(SystemExit):
                main(["--s5-exit", "break-even", "--only", "impulse_retest", "--jarvis-replay"])
            load.assert_not_called()

    def test_offline_requires_explicit_equity(self):
        with patch.object(launch, "replay") as call:
            with self.assertRaises(SystemExit):
                launch.main(["s5", "90", "--database", "archive.sqlite3"])
            call.assert_not_called()
