"""Streamlit operator dashboard. Run via `launch_dashboard.cmd`."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from importlib import invalidate_caches
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
root_text = str(ROOT)
if root_text in sys.path:
    sys.path.remove(root_text)
sys.path.insert(0, root_text)
for module_name in ("config.loader", "config.schema", "config"):
    sys.modules.pop(module_name, None)
invalidate_caches()

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.loader import PACKAGE_ROOT, load_credentials, load_settings, terminal_path_from_env
from core.instrument import AssetClass
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from dashboard.service import PROFILE_TIMEFRAMES, DashboardService, catalogue_asset_class
from infra.killswitch import KillSwitch
from reporting.pdf_report import build_pdf_report

OVERLAY = PACKAGE_ROOT / "config" / "eightcap.yaml"

st.set_page_config(page_title="MT5 Control Deck", page_icon="📈", layout="wide")
st.title("MT5 Control Deck")
st.caption("Read-only market intelligence and hard-stop control. Live execution is locked.")

settings = load_settings(overlay=OVERLAY)
credentials = load_credentials(required=False)
connector = MT5Connector(
    settings.mt5,
    credentials,
    terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
)
service = DashboardService(connector, settings)
kill_switch = KillSwitch.in_dir(PACKAGE_ROOT, settings.system.kill_switch_file)

try:
    account = service.connect()
    positions = service.positions()
    catalogue = service.symbols()
except Exception as exc:  # noqa: BLE001 - dashboard renders diagnostics instead of crashing
    st.error(f"MT5 connection failed: {exc}")
    st.stop()

try:
    asset_options = [asset.value for asset in AssetClass if asset is not AssetClass.UNKNOWN]
    selected_asset = st.sidebar.selectbox("Asset class", asset_options, index=0)
    candidates = [
        item for item in catalogue if catalogue_asset_class(item.path).value == selected_asset
    ]
    candidates.sort(key=lambda item: item.name)
    if not candidates:
        st.sidebar.warning(f"No {selected_asset} instruments reported by this broker account.")
        st.stop()
    preferred_symbol = {
        "forex": "EURUSD.i",
        "crypto": "BTCUSD",
        "stock": "AAPL",
        "index": "SPX500",
        "metal": "XAUUSD",
        "commodity": "USOUSD",
    }.get(selected_asset)
    default_index = next(
        (index for index, item in enumerate(candidates) if item.name == preferred_symbol), 0
    )
    selected_name = st.sidebar.selectbox(
        "Instrument",
        [item.name for item in candidates],
        index=default_index,
    )
    selected_descriptor = next(item for item in candidates if item.name == selected_name)
    spec = service.spec(selected_name)
    tick = service.tick(selected_name)

    defaults = [tf.value for tf in PROFILE_TIMEFRAMES[spec.asset_class][:3]]
    timeframe_names = st.sidebar.multiselect(
        "Charts (max 4)",
        [tf.value for tf in Timeframe],
        default=defaults,
        max_selections=4,
    )
    bar_count = st.sidebar.slider("Bars per chart", 100, 1000, 300, 50)
    if st.sidebar.button("Refresh broker data", use_container_width=True):
        st.rerun()

    first, second, third, fourth = st.columns(4)
    first.metric("Balance", f"{account.balance:.2f} {account.currency}")
    second.metric("Equity", f"{account.equity:.2f} {account.currency}")
    third.metric("Open positions", str(len(positions)))
    fourth.metric("Execution", "LOCKED" if not settings.mode.is_live else settings.mode.value)

    overview_tab, charts_tab, positions_tab, report_tab, control_tab = st.tabs(
        ["Overview", "Charts", "Positions", "PDF report", "Control"]
    )

    frames: dict[str, pd.DataFrame] = {}
    for name in timeframe_names:
        try:
            frames[name] = service.bars(selected_name, Timeframe.parse(name), bar_count)
        except Exception as exc:  # noqa: BLE001 - one unavailable timeframe must not kill the UI
            st.sidebar.warning(f"{name}: {exc}")

    with overview_tab:
        st.subheader(f"{selected_name} — {selected_descriptor.description}")
        st.write(
            {
                "asset_class": spec.asset_class.value,
                "broker_path": selected_descriptor.path,
                "minimum_lot": spec.volume_min,
                "lot_step": spec.volume_step,
                "contract_size": spec.contract_size,
                "trade_mode": spec.trade_mode,
            }
        )
        if tick is None:
            st.warning("No executable quote. The market may be closed; entry must remain blocked.")
        else:
            spread_pips = spec.price_to_pips(tick.spread)
            spread_bps = tick.spread / tick.mid * 10_000
            spread_cost = spec.money_per_lot(tick.spread) * spec.volume_min
            tick_age = max(0.0, (account.taken_at - tick.time).total_seconds())
            max_age = settings.filters.spread.max_tick_age_seconds.get(spec.asset_class.value)
            max_bps = settings.filters.spread.max_spread_bps.get(spec.asset_class.value)
            if max_age is None or tick_age > max_age:
                st.warning(
                    f"Stale quote: {tick_age:.0f}s old. It is displayed for context but "
                    "cannot clear the entry filter."
                )
            if max_bps is not None and spread_bps > max_bps:
                st.error(
                    f"Entry blocked: spread {spread_bps:.3f} bps exceeds the "
                    f"{max_bps:.3f} bps limit for {spec.asset_class.value}."
                )
            elif max_bps is not None:
                st.success(
                    f"Absolute spread gate clear: {spread_bps:.3f} / {max_bps:.3f} bps. "
                    "The adaptive baseline and all other risk gates must still pass."
                )
            a, b, c, d, e = st.columns(5)
            a.metric("Bid", f"{tick.bid:g}")
            b.metric("Ask", f"{tick.ask:g}")
            c.metric("Spread", f"{spread_pips:.2f} pips / {spread_bps:.3f} bps")
            d.metric("Min-lot spread cost", f"{spread_cost:.4f} {account.currency}")
            e.metric("Quote age", f"{tick_age:.0f}s")

    with charts_tab:
        if not frames:
            st.info("Select at least one available timeframe.")
        for timeframe, frame in frames.items():
            st.markdown(f"#### {selected_name} · {timeframe}")
            figure = go.Figure(
                data=[
                    go.Candlestick(
                        x=frame.index,
                        open=frame["open"],
                        high=frame["high"],
                        low=frame["low"],
                        close=frame["close"],
                        increasing_line_color="#22c55e",
                        decreasing_line_color="#ef4444",
                    )
                ]
            )
            figure.update_layout(
                height=430,
                margin={"l": 10, "r": 10, "t": 25, "b": 10},
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
            )
            st.plotly_chart(figure, use_container_width=True)

    with positions_tab:
        if not positions:
            st.success("No open positions.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ticket": p.ticket,
                            "symbol": p.symbol,
                            "side": p.direction.name,
                            "lots": p.volume,
                            "open": p.price_open,
                            "sl": p.sl,
                            "tp": p.tp,
                            "pnl": p.profit + p.swap,
                            "opened_utc": p.opened_at,
                        }
                        for p in positions
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    with report_tab:
        st.write("The report contains account state, positions, spread costs and selected charts.")
        pdf = build_pdf_report(account, positions, selected_name, spec, tick, frames)
        st.download_button(
            "Download PDF report",
            data=pdf,
            file_name=f"mt5-report-{selected_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with control_tab:
        pid_path = ROOT / "runtime" / "jarvis.pid"
        heartbeat_path = ROOT / "runtime" / "heartbeat.json"
        running_pid = int(pid_path.read_text().strip()) if pid_path.exists() else 0
        running = False
        if running_pid:
            if sys.platform == "win32":
                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, running_pid)
                running = bool(handle)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
            else:
                try:
                    os.kill(running_pid, 0)
                    running = True
                except OSError:
                    running = False
            if not running:
                pid_path.unlink(missing_ok=True)
        st.metric("Jarvis service", f"RUNNING (PID {running_pid})" if running else "OFF")
        if heartbeat_path.exists():
            st.json(heartbeat_path.read_text(encoding="utf-8"), expanded=False)
        stopped = kill_switch.is_engaged()
        st.metric("Hard STOP", "ENGAGED" if stopped else "clear")
        st.warning(
            "MONITOR and PAPER are autonomous. LIVE remains locked until the registered "
            "out-of-sample, demo and account-arming gates have all passed."
        )
        if st.button("STOP BOT NOW", type="primary", use_container_width=True):
            kill_switch.engage("operator dashboard emergency stop")
            st.rerun()
        start_monitor, start_paper = st.columns(2)
        creation_flags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        if start_monitor.button(
            "Start MONITOR", disabled=running or stopped, use_container_width=True
        ):
            subprocess.Popen(
                [sys.executable, str(ROOT / "jarvis.py"), "--operation", "monitor"],
                cwd=ROOT,
                creationflags=creation_flags,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            st.rerun()
        if start_paper.button("Start PAPER", disabled=running or stopped, use_container_width=True):
            subprocess.Popen(
                [sys.executable, str(ROOT / "jarvis.py"), "--operation", "paper"],
                cwd=ROOT,
                creationflags=creation_flags,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            st.rerun()
        confirmation = st.text_input("Type CLEAR STOP to reset the hard stop")
        if st.button(
            "Clear STOP",
            disabled=confirmation != "CLEAR STOP",
            use_container_width=True,
        ):
            kill_switch.clear()
            st.rerun()
finally:
    service.close()
