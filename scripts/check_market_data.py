"""Read-only check of the three affected markets using live data validation."""
from pathlib import Path

from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.clock import LiveClock
from core.data_manager import DataManager, _gap_boundaries, _missing_bars
from core.mt5_connector import MT5Connector
from core.types import Timeframe

ROOT = Path(__file__).resolve().parent.parent


def main():
    settings = load_settings(overlay=ROOT / 'config/eightcap.yaml')
    broker = MT5Connector(settings.mt5, load_credentials(required=True),
                          terminal_path=settings.mt5.terminal_path or terminal_path_from_env())
    broker.connect()
    failed = False
    try:
        manager = DataManager(broker, settings.data, LiveClock())
        print(f'Broker timestamp offset: {broker.server_offset}')
        for symbol in ('NDX100', 'XAUUSD', 'SPX500'):
            print(f'\n{symbol}')
            try:
                count = settings.data.bars['M5']
                frame = manager._drop_forming_bar(manager._to_frame(
                    broker.copy_rates(symbol, Timeframe.M5.mt5_value, count + 2)), Timeframe.M5).iloc[-count:]
                print(f'M5 closed bars: {len(frame)}; largest gap boundaries UTC: {_gap_boundaries(frame, Timeframe.M5)}')
                closures = settings.data.history_closures.get(symbol, ())
                print(f'Missing before calendar: {_missing_bars(frame, Timeframe.M5):.0f}; '
                      f'after calendar: {_missing_bars(frame, Timeframe.M5, closures=closures):.0f}')
                manager._validate(symbol, Timeframe.M5, frame)
                manager.get_context(symbol)
                print('PASS: live market context available')
            except Exception as exc:
                failed = True
                print(f'FAIL: {type(exc).__name__}: {exc}')
    finally:
        broker.shutdown()
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
