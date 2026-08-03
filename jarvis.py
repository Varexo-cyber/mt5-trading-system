"""CLI for the autonomous scanner/trader service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from infra.logging import setup_logging
from runner.profiles import PROFILES, apply_profile
from runner.service import JarvisRunner, OperationMode

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation", choices=[mode.value for mode in OperationMode], default="monitor"
    )
    parser.add_argument("--once", action="store_true", help="Run one bounded scan cycle")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        help="; ".join(f"{name}: {p.description}" for name, p in sorted(PROFILES.items())),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    if args.profile:
        settings = apply_profile(settings, args.profile)
    setup_logging(
        level=settings.logging.level,
        log_dir=ROOT / settings.logging.directory,
        filename=settings.logging.filename,
        console=settings.logging.console,
        console_level=settings.logging.console_level,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
    )
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    if args.profile:
        profile = PROFILES[args.profile]
        print(
            f"profile '{profile.name}': {profile.description}\n"
            f"  markets  : {', '.join(profile.symbols_only or profile.asset_classes) or 'all'}\n"
            f"  positions: {settings.effective_max_positions()}"
        )
    runner = JarvisRunner(connector, settings, ROOT, OperationMode(args.operation))
    pid_path = ROOT / "runtime" / "jarvis.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        if args.once:
            runner.connect()
            try:
                summary = runner.run_once()
                print(summary)
            finally:
                runner.close()
        else:
            runner.run_forever()
    finally:
        pid_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
