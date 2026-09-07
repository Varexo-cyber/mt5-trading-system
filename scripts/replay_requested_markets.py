"""Operator launchers using the unmodified detector path of hoeveel.cmd."""
from __future__ import annotations

import argparse

from scripts.dry_run_sections import main as replay


def commands(kind: str, days: int, database: str = "", equity: float = 0.0):
    common = ["--days", str(days), "--jarvis-replay"]
    if database:
        common += ["--database", database]
    if equity:
        common += ["--equity", str(equity)]
    if kind == "us30":
        return [common + ["--symbols", "US30", "--only", "impulse_retest,order_block_fast",
                          "--sweep", tf, "--csv", f"runtime/us30-{tf.lower()}-{days}.csv"]
                for tf in ("M1", "M5")]
    return [common + ["--symbols", "NDX100", "--only", "section_five_ndx100_m5",
                      "--s5-exit", mode, "--csv", f"runtime/ndx100-s5-{mode}-{days}.csv"]
            for mode in ("fixed", "break-even")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("us30", "s5"))
    parser.add_argument("days", type=int, nargs="?", default=180)
    parser.add_argument("--database", default="")
    parser.add_argument("--equity", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.days <= 0:
        parser.error("days must be positive")
    if args.database and args.equity <= 0:
        parser.error("offline database replay requires --equity")
    for index, command in enumerate(commands(args.kind, args.days, args.database, args.equity), 1):
        print(f"\nMEASUREMENT {index}/2: {' '.join(command)}", flush=True)
        replay(command)


if __name__ == "__main__":
    main()
