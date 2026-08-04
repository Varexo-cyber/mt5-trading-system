"""Streamlit operator dashboard. Run via `launch_dashboard.cmd`."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
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

from advisory import read_recent_reviews
from advisory.veto_memory import VetoMemory
from config.loader import PACKAGE_ROOT, load_credentials, load_settings, terminal_path_from_env
from core.instrument import AssetClass
from core.mt5_connector import MT5Connector
from core.types import Direction, Timeframe
from dashboard.ai_exchange import (
    pair_ai_reviews,
    read_block_reason,
    read_posture,
    supervision_rows,
)
from dashboard.ledger import as_rows, day_start, recent_management, summarise, week_start
from dashboard.position_control import PositionControl
from dashboard.service import (
    PROFILE_TIMEFRAMES,
    DashboardService,
    catalogue_asset_class,
    load_paper_snapshot,
)
from infra.killswitch import KillSwitch
from learning.memory import TradingMemory
from monitoring.scan_activity import read_scan_activity
from promotion.experimental import (
    ExperimentalLiveContract,
    apply_experimental_live_limits,
    contract_path,
)
from reporting.pdf_report import build_pdf_report

OVERLAY = PACKAGE_ROOT / "config" / "eightcap.yaml"

st.set_page_config(page_title="MT5 Control Deck", page_icon="📈", layout="wide")
st.title("MT5 Control Deck")
st.caption("Market intelligence, reporting and hard-stop control for Jarvis.")

settings = load_settings(overlay=OVERLAY)
credentials = load_credentials(required=False)
connector = MT5Connector(
    settings.mt5,
    credentials,
    terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
)
service = DashboardService(connector, settings)
kill_switch = KillSwitch.in_dir(PACKAGE_ROOT, settings.system.kill_switch_file)
ai_ready = (
    settings.ai.enabled
    and settings.ai.provider == "anthropic"
    and bool(settings.ai.anthropic_model)
    and bool(os.getenv("ANTHROPIC_API_KEY"))
)


@st.fragment(run_every="5s")
def render_live_scanner() -> None:
    state = read_scan_activity(ROOT / "runtime" / "scan_activity.json")
    if not state:
        st.info(
            "Nog geen scanlog beschikbaar. Wis STOP en start een Jarvis-modus; "
            "de volledige ondersteunde Eightcap-catalogus verschijnt daarna hier."
        )
        return

    symbols = list(state.get("symbols", {}).values())
    recent = list(state.get("recent", []))
    universe_size = int(state.get("universe_size", 0))
    seen = len(symbols)
    coverage = min(100.0, 100.0 * seen / universe_size) if universe_size else 0.0
    last_batch = state.get("last_batch", {})
    with st.container(horizontal=True):
        st.metric("Broker-markten", f"{universe_size:,}", border=True)
        st.metric("Unieke markten gezien", f"{seen:,}", border=True)
        st.metric("Rotatie-dekking", f"{coverage:.1f}%", border=True)
        st.metric(
            "Laatste batch",
            f"{int(last_batch.get('inspected', 0))} bekeken",
            border=True,
        )
        st.metric("Totaal inspecties", f"{int(state.get('total_inspections', 0)):,}", border=True)

    st.caption(
        f"Automatische update iedere 5 seconden · scannerstand: {state.get('operation', 'off')} · "
        f"laatst bijgewerkt: {state.get('updated_at', 'onbekend')}"
    )
    # Repeated here as well as at the top of the page, because this is the tab
    # an operator opens to ask "is it doing anything" and the counters alone
    # cannot answer it: a fully halted system scans exactly as busily as a
    # working one.
    halted = read_block_reason(ROOT / "runtime" / "heartbeat.json")
    if halted:
        st.error(
            f"**Er wordt niets diep geanalyseerd: {halted['reason']}**\n\n"
            f"{halted['detail']}\n\nDe scan hieronder loopt gewoon door — het is de "
            "vervolgstap die wordt overgeslagen zolang er toch niet gehandeld mag worden."
        )
    st.info(
        "Iedere cyclus screent de volledige ondersteunde Eightcap-catalogus. "
        "De doeltijd tussen cycli is 30 seconden, maar een volledige scan kan langer duren; "
        "dan begint de volgende cyclus zodra MT5 klaar is. Alleen de best gerangschikte "
        "markten krijgen daarna de zware multi-timeframeanalyse."
    )

    view = st.segmented_control(
        "Weergave",
        ["Recente inspecties", "Laatste status per markt"],
        default="Recente inspecties",
        key="scanner_view",
    )
    rows = recent if view == "Recente inspecties" else symbols
    asset_classes = sorted({str(row.get("asset_class", "unknown")) for row in rows})
    selected_classes = (
        st.pills(
            "Assetklassen",
            asset_classes,
            default=asset_classes,
            selection_mode="multi",
            key="scanner_asset_classes",
        )
        or []
    )
    filtered = [row for row in rows if row.get("asset_class") in selected_classes]
    display_rows = []
    for row in reversed(filtered[-250:]):
        deep_status = row.get("deep_status")
        deep_reason = row.get("deep_reason")
        deep_detail = row.get("deep_detail")
        why = (
            f"{deep_reason}: {deep_detail}"
            if deep_reason and deep_detail
            else deep_reason or deep_detail or row.get("reason")
        )
        display_rows.append(
            {
                "Tijd (UTC)": row.get("deep_at") or row.get("inspected_at"),
                "Markt": row.get("symbol"),
                "Klasse": row.get("asset_class"),
                "Besluit": deep_status or row.get("status"),
                "Fase": "deep analysis" if deep_status else row.get("stage"),
                "Waarom": why,
                "Spread (bps)": row.get("spread_bps"),
                "Quoteleeftijd (s)": row.get("quote_age_seconds"),
                "Rangscore": row.get("rank"),
            }
        )
    if display_rows:
        st.dataframe(
            pd.DataFrame(display_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Spread (bps)": st.column_config.NumberColumn(format="%.3f"),
                "Quoteleeftijd (s)": st.column_config.NumberColumn(format="%.1f"),
                "Rangscore": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        reason_counts = (
            pd.DataFrame(display_rows)["Besluit"]
            .value_counts()
            .rename_axis("Besluit")
            .reset_index(name="Aantal")
        )
        st.bar_chart(reason_counts, x="Besluit", y="Aantal")
    else:
        st.warning("Geen scanregels voor deze selectie.")

    with st.expander("Hoe Jarvis van scan naar trade gaat"):
        st.markdown("""
1. Haal alle ondersteunde symbolen uit de Eightcap-catalogus.
2. Controleer per symbool of het verhandelbaar is en een verse prijs heeft.
3. Blokkeer een te hoge spread of ontbrekende H1-geschiedenis.
4. Geef alle overblijvers een lichte trend/activiteit-rangscore.
5. Analyseer maximaal vijf winnaars zwaar op D1, H4, H1, M15 en M5.
6. Controleer whitelist, balans, bestaande posities, nieuws, sessie en correlatie.
7. Bereken een echte stoploss, take-profit, lotgrootte en maximaal 1% risico.
8. Vraag Claude als laatste veto. Claude mag nooit risico of orderwaarden veranderen.
9. Stuur alleen bij alle groene poorten een order naar MT5/Eightcap.
10. Beheer daarna SL, break-even, trailing/exit en schrijf alles in het journaal.
""")


@st.fragment(run_every="1s")
def render_account_header(operation: str, paper) -> None:  # type: ignore[no-untyped-def]
    """Balance and equity, re-read from the terminal every second.

    These were rendered once at page load and then frozen. Equity moves with
    every tick on an open position, so the headline number an operator glances
    at was arbitrarily old — it only refreshed when something else happened to
    rerun the page. On a live account the top-line figure has to be the current
    one or it is worse than absent.
    """
    try:
        account = service.account()
        open_count = len(service.positions())
    except Exception:  # noqa: BLE001 - a dropped link must not blank the header
        st.warning("Terminal even niet bereikbaar — cijfers hieronder zijn de laatst bekende.")
        account = service.account_snapshot
        open_count = 0
        if account is None:
            return

    first, second, third, fourth, fifth = st.columns(5)
    first.metric("Balance", f"{account.balance:.2f} {account.currency}")
    floating = account.equity - account.balance
    second.metric(
        "Equity",
        f"{account.equity:.2f} {account.currency}",
        delta=f"{floating:+.2f}" if abs(floating) >= 0.005 else None,
    )
    third.metric("Open positions", str(open_count))
    fourth.metric("Jarvis mode", operation)
    fifth.metric(
        "Paper equity",
        f"{paper.equity:.2f} {paper.currency}" if paper is not None else "not started",
    )
    st.caption(f"Live · {datetime.now(UTC):%H:%M:%S} UTC")


@st.fragment(run_every="2s")
def render_trade_history() -> None:
    """What today and this week actually did — the view that did not exist.

    The deck showed what was open and what was being considered, and nothing
    about what had already happened. Which stop was hit, what today cost, where
    the day started: all of it lived only in the journal, which nobody reads
    mid-session.
    """
    database = ROOT / settings.journal.database_path
    now = datetime.now(UTC)
    boundary = settings.risk.day_boundary_utc
    opened_today = day_start(now, boundary)
    opened_week = week_start(now, boundary)
    today = summarise(database, "Vandaag", opened_today, "DAY", opened_today)
    week = summarise(database, "Deze week", opened_week, "WEEK", opened_week)

    st.subheader("Wat er vandaag en deze week is gebeurd")
    for period in (today, week):
        st.markdown(f"**{period.label}** — vanaf {period.started_at:%d-%m %H:%M} UTC")
        a, b, c, d, e = st.columns(5)
        a.metric("Trades gesloten", len(period.trades))
        b.metric(
            "Resultaat",
            f"{period.realised:+.2f}",
            delta=f"{period.total_r:+.2f}R" if period.trades else None,
        )
        c.metric("Gewonnen", period.wins)
        d.metric("Verloren", period.losses)
        e.metric("Winratio", f"{period.win_rate:.0%}" if period.trades else "—")
        if period.starting_equity is not None:
            st.caption(f"Begonnen met {period.starting_equity:.2f} EUR")
        if period.trades:
            best, worst = period.best, period.worst
            if best is not None and worst is not None and best is not worst:
                st.caption(
                    f"Beste: {best.symbol} {best.pnl_money:+.2f}  ·  "
                    f"Slechtste: {worst.symbol} {worst.pnl_money:+.2f}"
                )
        st.write("")

    if not today.trades and not week.trades:
        st.info(
            "Nog geen afgesloten trades in het journaal. Posities die je zelf in de terminal "
            "hebt geopend of gesloten staan hier niet: dit toont wat het systeem zelf heeft "
            "gedaan, zodat je kunt beoordelen hoe *het* presteert."
        )
        return

    st.markdown("**Alle afgesloten trades deze week**")
    frame = pd.DataFrame(as_rows(week.trades))
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Resultaat": st.column_config.NumberColumn(format="%.2f"),
            "R": st.column_config.NumberColumn(format="%.2f"),
            "Entry": st.column_config.NumberColumn(format="%.5f"),
            "Exit": st.column_config.NumberColumn(format="%.5f"),
        },
    )

    # How trades end is the fastest read on whether management is working: a
    # wall of stop losses and a wall of targets need opposite responses.
    endings = frame["Hoe het eindigde"].value_counts().rename_axis("Einde").reset_index(name="n")
    if len(endings) > 1:
        st.bar_chart(endings, x="Einde", y="n")


@st.fragment(run_every="2s")
def render_management_log() -> None:
    """Every stop move, trail and protective exit the guard has made.

    The guard runs roughly once a second between cycles. Without this panel it
    is indistinguishable from a system doing nothing — and "is it actually
    watching my positions" is the question this whole layer exists to answer.
    """
    rows = recent_management(ROOT / settings.journal.database_path, limit=40)
    st.subheader("Wat Jarvis met de posities heeft gedaan")
    if not rows:
        st.caption(
            "Nog geen beheeracties. Deze verschijnen zodra een positie break-even bereikt, "
            "de stop wordt meegetrokken, of winst wordt veiliggesteld."
        )
        return
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"R": st.column_config.NumberColumn(format="%.2f")},
    )


def render_recent_adoptions() -> None:
    """Say when a position was recovered rather than closed after a crash.

    Adoption is the right outcome but it is not a normal one, and it happening
    silently would hide the fact that the process died mid-entry. The operator
    should know a restart cost them nothing — and equally, that a restart
    happened at all.
    """
    database = ROOT / settings.journal.database_path
    if not database.exists():
        return
    try:
        import sqlite3

        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT a.ts, a.note, t.symbol, t.ticket FROM management_actions a "
                "JOIN trades t ON t.id = a.trade_id WHERE a.action = 'ADOPTED' "
                "ORDER BY a.id DESC LIMIT 5"
            ).fetchall()
            pending = conn.execute(
                "SELECT symbol, direction, volume, opened_at FROM trades "
                "WHERE entry_state = 'PENDING' AND closed_at IS NULL ORDER BY opened_at"
            ).fetchall()
    except sqlite3.Error:
        # A locked or half-migrated journal must not take the panel down; the
        # positions below are the part that matters.
        return

    for row in pending:
        st.warning(
            f"Openstaande order-intentie: {row['symbol']} {row['direction']} "
            f"{row['volume']:g} lots ({row['opened_at']}). Als de broker deze positie "
            "wél heeft geopend, adopteert Jarvis hem bij de volgende cyclus."
        )
    if rows:
        with st.expander(f"Herstelde posities na een crash ({len(rows)})"):
            st.caption(
                "Deze posities stonden bij de broker open zonder journaalregel, en zijn "
                "gekoppeld aan de order-intentie die ze had aangemaakt in plaats van "
                "gesloten te worden."
            )
            for row in rows:
                st.markdown(f"- `{row['ts']}` **{row['symbol']}** #{row['ticket']} — {row['note']}")


@st.fragment(run_every="1s")
def render_live_positions(account) -> None:  # type: ignore[no-untyped-def]
    """Live open positions with per-position control.

    A one-second fragment rather than a whole-page refresh: this is the one
    panel where the numbers are money moving in real time, and re-running the
    entire dashboard that often would re-fetch the catalogue and every chart.
    The fragment re-reads only what it draws.

    Everything shown is pulled fresh from the terminal on each pass — the
    position list, the account, and a tick per symbol. Nothing here is cached
    or carried over from the page load, because a stale P&L on a live position
    is worse than no P&L: it invites a decision based on a price that has
    already moved.

    Every control can be exercised while Jarvis is running. The two do not
    conflict — MT5 is the single source of truth for what is open, so a stop the
    operator moves is simply the stop the engine sees on its next cycle.
    """
    control = PositionControl(connector, settings)
    # A read failure here is a display problem, not a trading one, and it must
    # look like one. MT5 drops its IPC channel whenever the terminal restarts
    # or times the client out, and the resulting exception arrived as a full
    # Streamlit traceback where the positions table should have been — on an
    # account holding real money, which is the worst possible moment to be
    # shown a stack trace instead of your positions.
    try:
        positions = service.positions()
        account = service.account()
    except Exception as exc:  # noqa: BLE001 - any read failure degrades the same way
        st.error(
            f"**Kan de posities nu niet uitlezen** — {type(exc).__name__}\n\n"
            f"{exc}\n\nDit is alleen het dashboard. Jarvis heeft zijn eigen verbinding en "
            "blijft je posities beheren. Meestal is de MT5-terminal net herstart of even "
            "weggevallen; de volgende verversing pakt hem vanzelf weer op."
        )
        return
    notice = st.session_state.pop("position_notice", None)
    if notice:
        (st.success if notice[0] else st.error)(notice[1])

    render_recent_adoptions()

    if not positions:
        st.success("No open positions.")
        return

    total = sum(p.profit + p.swap for p in positions)
    exposure = sum(p.volume for p in positions)
    a, b, c = st.columns(3)
    a.metric("Open positions", len(positions))
    b.metric("Floating P&L", f"{total:+.2f} {account.currency}")
    c.metric("Total lots", f"{exposure:g}")

    st.caption(
        f"Live, ververst iedere seconde · {datetime.now(UTC):%H:%M:%S} UTC. SL/TP aanpassen en "
        "sluiten kan terwijl Jarvis draait — MT5 is de bron van waarheid, dus Jarvis ziet je "
        "wijziging bij de volgende cyclus. Een SL strakker zetten mag altijd; ruimer zetten "
        f"wordt geweigerd zodra het risico boven {settings.effective_max_risk_pct():.2f}% van "
        "je equity uitkomt."
    )

    for position in positions:
        tick = service.tick(position.symbol)
        price = 0.0
        if tick is not None:
            price = tick.bid if position.direction is Direction.LONG else tick.ask
        pnl = position.profit + position.swap
        risk = abs(position.price_open - position.sl) if position.sl else 0.0
        r_now = (
            ((price - position.price_open) * int(position.direction) / risk)
            if (risk and price)
            else None
        )
        header = (
            f"{'🟢' if pnl >= 0 else '🔴'} #{position.ticket} · {position.symbol} · "
            f"{position.direction.name} {position.volume:g} lots · {pnl:+.2f} "
            f"{account.currency}" + (f" · {r_now:+.2f}R" if r_now is not None else "")
        )
        with st.expander(header, expanded=len(positions) <= 3):
            # Where price sits between the stop and the target, right now. The
            # number that matters on a live trade is not the price, it is how
            # much room is left in each direction.
            if price and position.sl and position.tp:
                span = abs(position.tp - position.sl)
                travelled = abs(price - position.sl)
                if span > 0:
                    st.progress(
                        min(1.0, max(0.0, travelled / span)),
                        text=(
                            f"stop {abs(price - position.sl):.5g} weg  ·  "
                            f"target {abs(position.tp - price):.5g} weg"
                        ),
                    )
            cols = st.columns(5)
            cols[0].metric("Entry", f"{position.price_open:g}")
            cols[1].metric("Now", f"{price:g}" if price else "—")
            cols[2].metric("Stop", f"{position.sl:g}" if position.sl else "NONE")
            cols[3].metric("Target", f"{position.tp:g}" if position.tp else "none")
            cols[4].metric("Age", _age(position.opened_at))

            if not position.has_stop:
                st.error(
                    "This position has no stop loss. Jarvis closes an unprotected position "
                    "on its next cycle; set one here if you want to keep it."
                )

            spec = service.spec(position.symbol)
            step = 10.0**-spec.digits
            edit, act = st.columns([3, 2])
            with edit, st.form(f"modify-{position.ticket}"):
                st.markdown("**Stop en target aanpassen**")
                new_sl = st.number_input(
                    "Stop loss",
                    value=float(position.sl),
                    step=step,
                    format=f"%.{spec.digits}f",
                    key=f"sl-{position.ticket}",
                )
                new_tp = st.number_input(
                    "Take profit (0 = geen)",
                    value=float(position.tp),
                    step=step,
                    format=f"%.{spec.digits}f",
                    key=f"tp-{position.ticket}",
                )
                preview = control.preview_stop(position, new_sl, account.equity)
                if not preview.valid or not preview.permitted:
                    st.warning(preview.detail)
                else:
                    st.caption(preview.detail)
                if st.form_submit_button("Wijzig SL/TP", width="stretch"):
                    outcome = control.modify(position, sl=new_sl, tp=new_tp, equity=account.equity)
                    st.session_state["position_notice"] = (outcome.ok, outcome.message)
                    st.rerun(scope="fragment")

            with act:
                st.markdown("**Sluiten**")
                if st.button(
                    f"Sluit #{position.ticket} volledig",
                    key=f"close-{position.ticket}",
                    width="stretch",
                    type="primary",
                ):
                    outcome = control.close(position)
                    st.session_state["position_notice"] = (outcome.ok, outcome.message)
                    st.rerun(scope="fragment")
                half = spec.round_volume_down(position.volume / 2)
                if half >= spec.volume_min and position.volume - half >= spec.volume_min:
                    if st.button(
                        f"Sluit de helft ({half:g} lots)",
                        key=f"half-{position.ticket}",
                        width="stretch",
                    ):
                        outcome = control.close(position, half)
                        st.session_state["position_notice"] = (outcome.ok, outcome.message)
                        st.rerun(scope="fragment")
                else:
                    st.caption(
                        f"Deels sluiten kan niet: {position.volume:g} lots is niet te splitsen "
                        f"boven het minimum van {spec.volume_min:g}."
                    )
                in_profit = (
                    bool(position.sl)
                    and bool(price)
                    and (price - position.price_open) * int(position.direction) > 0
                )
                if in_profit and st.button(
                    "Stop naar break-even",
                    key=f"be-{position.ticket}",
                    width="stretch",
                ):
                    outcome = control.modify(
                        position,
                        sl=position.price_open,
                        tp=position.tp,
                        equity=account.equity,
                    )
                    st.session_state["position_notice"] = (outcome.ok, outcome.message)
                    st.rerun(scope="fragment")

    st.divider()
    with st.expander("Alles sluiten"):
        st.warning(
            "Dit sluit iedere open positie op dit account. Het is géén STOP: Jarvis blijft "
            "draaien en mag daarna nieuwe trades openen. Gebruik de Control-tab als je het "
            "systeem echt wilt stilzetten."
        )
        confirmed = (
            st.text_input("Typ SLUIT ALLES om te bevestigen", key="close-all-confirm").strip()
            == "SLUIT ALLES"
        )
        if confirmed and st.button("Sluit alle posities", type="primary"):
            results = control.close_all(positions)
            failed = [item for item in results if not item.ok]
            st.session_state["position_notice"] = (
                not failed,
                " | ".join(item.message for item in results) or "Niets te sluiten.",
            )
            st.rerun(scope="fragment")


def _age(opened_at) -> str:  # type: ignore[no-untyped-def]
    seconds = max(0.0, (datetime.now(UTC) - opened_at).total_seconds())
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


@st.fragment(run_every="5s")
def render_learning() -> None:
    """What the account has taught itself, and what it is currently refusing.

    Both of these used to be invisible. Lessons were paid for and discarded, and
    a suppressed veto looked identical to a market that simply produced no
    signal — so "why has it not looked at SPX500 for an hour" had no answer
    anywhere in the interface.
    """
    memory = TradingMemory(ROOT / "runtime" / "trading_memory.json")
    vetoes = VetoMemory(ROOT / "runtime" / "veto_memory.json")
    now = datetime.now(UTC)
    brief = memory.briefing()

    a, b, c = st.columns(3)
    a.metric("Afgesloten trades geleerd", brief["closed_trades_recorded"])
    b.metric("Cumulatief resultaat", f"{brief['cumulative_r']:+.2f}R")
    c.metric("Actieve veto's", len(vetoes.active(now)))

    stance = read_posture(ROOT / "runtime" / "heartbeat.json")
    if stance and stance.get("posture") != "steady":
        st.warning(
            f"**Houding: {str(stance['posture']).upper()}** — "
            f"{stance.get('consecutive_losses', 0)} verliezen op rij, "
            f"{stance.get('drawdown_from_peak_pct', 0):.1f}% onder de piek. "
            "Verliezende posities worden sneller gesloten en de lat voor een nieuwe trade "
            "ligt hoger. De positiegrootte verandert niet — die staat vast en gaat na "
            "verlies nooit omhoog."
        )
    elif stance:
        st.success("Houding: STEADY — normale werking, geen verliesreeks of drawdown.")

    st.markdown("**Lessen uit eigen trades** — gaan mee in elk volgend verzoek aan Claude.")
    lessons = brief["lessons"]
    if lessons:
        for lesson in lessons:
            st.markdown(f"- {lesson}")
    else:
        st.info(
            "Nog geen lessen. Deze verschijnen zodra er trades zijn afgesloten en Claude "
            "die heeft geëvalueerd."
        )

    worst = brief["worst_performing"]
    if worst:
        st.markdown("**Slechtst presterende markten**")
        for line in worst:
            st.markdown(f"- {line}")

    st.markdown("**Wat nu geweigerd blijft** — deze worden niet opnieuw naar Claude gestuurd.")
    active = vetoes.active(now)
    if active:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Markt": record.symbol,
                        "Richting": record.direction,
                        "Keer geweigerd": record.repeats,
                        "Confidence": record.confidence,
                        "Stil tot (UTC)": record.suppress_until,
                        "Reden": record.thesis,
                    }
                    for record in active
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={"Confidence": st.column_config.NumberColumn(format="%.2f")},
        )
    else:
        st.success("Geen enkele markt staat op dit moment op de geweigerd-lijst.")


@st.fragment(run_every="3s")
def render_supervision() -> None:
    """Claude's decisions about positions that are already open."""
    rows = supervision_rows(read_recent_reviews(ROOT / "runtime" / "ai_reviews.jsonl", limit=400))
    if not rows:
        st.info(
            "Nog geen beheerbeslissingen. Zodra er een positie openstaat beoordeelt Claude "
            f"die iedere {settings.trade_management.supervision_interval_minutes:g} minuten "
            "en kan hij hem strakker zetten, deels sluiten of helemaal sluiten."
        )
        return
    acted = [row for row in rows if row["action"] not in {"hold", "?"}]
    a, b, c = st.columns(3)
    a.metric("Beoordelingen", len(rows))
    b.metric("Ingegrepen", len(acted))
    c.metric("Vastgehouden", len(rows) - len(acted))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Tijd (UTC)": row["at"],
                    "Ticket": row["ticket"],
                    "Markt": row["symbol"],
                    "Richting": row["direction"],
                    "Actie": row["action"],
                    "Confidence": row["confidence"],
                    "Duur (ms)": row["latency_ms"],
                    "Claude zegt": row["reason"],
                }
                for row in rows
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Confidence": st.column_config.NumberColumn(format="%.2f"),
            "Duur (ms)": st.column_config.NumberColumn(format="%.1f"),
        },
    )


@st.fragment(run_every="3s")
def render_ai_exchange() -> None:
    review_rows = read_recent_reviews(ROOT / "runtime" / "ai_reviews.jsonl", limit=200)
    exchanges = pair_ai_reviews(review_rows)
    decisions = [item for item in exchanges if item["status"] != "PENDING"]
    approvals = sum(item["status"] == "APPROVED" for item in decisions)
    vetoes = sum(item["status"] == "VETO" for item in decisions)
    errors = sum(item["status"] == "ERROR / FAIL CLOSED" for item in decisions)
    pending = sum(item["status"] == "PENDING" for item in exchanges)

    with st.container(horizontal=True):
        st.metric("Naar Claude gestuurd", len(exchanges), border=True)
        st.metric("Goedgekeurd", approvals, border=True)
        st.metric("Veto", vetoes, border=True)
        st.metric("API-/auditfouten", errors, border=True)
        st.metric("Wacht op antwoord", pending, border=True)

    st.caption(
        "Automatische update iedere 3 seconden. Alleen voorstellen die analyse, risico, "
        "filters, sizing en marge al hebben gehaald, worden betaald naar Claude gestuurd."
    )

    if exchanges:
        summary_rows = []
        for item in exchanges:
            decision = item["decision"] if isinstance(item["decision"], dict) else {}
            summary_rows.append(
                {
                    "Verzoek (UTC)": item["requested_at"],
                    "Antwoord (UTC)": item["responded_at"],
                    "Markt": item["symbol"],
                    "Richting": item["direction"],
                    "Status": item["status"],
                    "Duur (ms)": item["latency_ms"],
                    "Confidence": decision.get("confidence"),
                    "Claude zegt": decision.get("thesis") or decision.get("error"),
                    "Risico's": ", ".join(str(risk) for risk in decision.get("risks", []) or []),
                }
            )
        st.dataframe(
            pd.DataFrame(summary_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Duur (ms)": st.column_config.NumberColumn(format="%.1f"),
                "Confidence": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.subheader("Exact verzoek en exact antwoord")
        for item in exchanges[:10]:
            label = (
                f"{item['status']} · {item['symbol']} {item['direction']} · "
                f"{item['requested_at'] or 'tijd onbekend'}"
            )
            with st.expander(label):
                request_column, response_column = st.columns(2)
                with request_column:
                    st.markdown("**Signaalsysteem → Claude**")
                    st.json(item["request"], expanded=False)
                with response_column:
                    st.markdown("**Claude → Jarvis**")
                    if item["decision"]:
                        st.json(item["decision"], expanded=False)
                    else:
                        st.warning("Claude heeft nog geen antwoord teruggegeven.")
    else:
        st.info(
            "Nog niets naar Claude gestuurd. Dat betekent niet dat Jarvis stilstaat: "
            "tot nu toe heeft geen kandidaat alle vaste poorten vóór Claude gehaald."
        )

    scan_state = read_scan_activity(ROOT / "runtime" / "scan_activity.json")
    latest_symbols = list(scan_state.get("symbols", {}).values()) if scan_state else []
    blocked_before_ai = sorted(
        (row for row in latest_symbols if row.get("deep_status") == "DEEP_REJECTED"),
        key=lambda row: str(row.get("deep_at", "")),
        reverse=True,
    )[:25]
    st.subheader("Wel diep bekeken, niet naar Claude gestuurd")
    if blocked_before_ai:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Tijd (UTC)": row.get("deep_at"),
                        "Markt": row.get("symbol"),
                        "Besluit": row.get("deep_reason"),
                        "Waarom": row.get("deep_detail"),
                        "Naar Claude": "Nee",
                    }
                    for row in blocked_before_ai
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("Nog geen diepe afwijzingen geregistreerd.")


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
    if st.sidebar.button("Refresh broker data", width="stretch"):
        st.rerun()

    heartbeat_path = ROOT / "runtime" / "heartbeat.json"
    try:
        heartbeat = (
            json.loads(heartbeat_path.read_text(encoding="utf-8"))
            if heartbeat_path.exists()
            else {}
        )
    except (OSError, json.JSONDecodeError):
        heartbeat = {}
    pid_path = ROOT / "runtime" / "jarvis.pid"
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
    experimental_contract = None
    experimental_error = "not armed"
    try:
        experimental_contract = ExperimentalLiveContract.load(contract_path(ROOT))
        experimental_contract.assert_compatible(
            account,
            apply_experimental_live_limits(settings),
        )
        experimental_error = ""
    except RuntimeError as exc:
        experimental_error = str(exc)
    paper = load_paper_snapshot(ROOT / "runtime" / "paper_state.json")
    active_operation = str(heartbeat.get("operation", "OFF")).upper() if running else "OFF"
    render_account_header(active_operation, paper)
    if running and heartbeat.get("operation") == "experimental_live":
        st.error(
            f"REAL MONEY ACTIVE - account {account.login}, equity {account.equity:.2f} "
            f"{account.currency}. Use the Control tab for the hard stop."
        )
    elif kill_switch.is_engaged():
        st.warning("Jarvis is OFF because the durable hard STOP is engaged.")

    # Why no new trades, at the top of the page rather than in one log line.
    #
    # A halted account and a broken one look identical from here: the scanner
    # keeps counting, the cycle log keeps scrolling, and every candidate shows
    # "0 analysed" with no reason given anywhere in the interface. That is the
    # state this deck exists to make legible, and it was the one thing it did
    # not show.
    blocked = str(heartbeat.get("blocked_reason", "") or "")
    if running and blocked:
        detail = str(heartbeat.get("blocked_detail", "") or "")
        st.error(
            f"**GEEN NIEUWE TRADES — {blocked}**\n\n{detail}\n\n"
            "Jarvis draait en scant nog volledig, en beheert bestaande posities gewoon door. "
            "Hij opent alleen niets nieuws zolang dit geldt. Daarom staat er 0 diep "
            "geanalyseerd: de analyse wordt overgeslagen zodra vaststaat dat er toch niet "
            "gehandeld mag worden."
        )
    stance = dict(heartbeat.get("posture") or {}) if running else {}
    if stance and stance.get("posture") not in {None, "steady"}:
        st.warning(
            f"**Houding: {str(stance['posture']).upper()}** — "
            f"{stance.get('consecutive_losses', 0)} verliezen op rij, "
            f"{stance.get('drawdown_from_peak_pct', 0):.1f}% onder de piek. "
            f"Per cyclus wordt alleen de beste {stance.get('candidates_allowed', 1)} setup "
            "opgepakt en verliezende posities worden sneller gesloten. De positiegrootte "
            "verandert niet."
        )

    overview_tab, scanner_tab, ai_tab, charts_tab, positions_tab, report_tab, control_tab = st.tabs(
        [
            "Overview",
            "Live scanner",
            "AI exchange",
            "Charts",
            "Positions",
            "PDF report",
            "Control",
        ]
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

    with scanner_tab:
        st.subheader("Wat Jarvis achter de schermen scant")
        st.warning(
            "De hele catalogus wordt bekeken, maar met EUR 100 mogen alleen EURUSD.i, "
            "GBPUSD.i, USDJPY.i en AUDUSD.i uiteindelijk een echte order worden."
        )
        render_live_scanner()

    with ai_tab:
        entry_view, manage_view, learn_view = st.tabs(
            ["Instapbeoordeling", "Beheer van open posities", "Wat het systeem heeft geleerd"]
        )
        with entry_view:
            st.subheader("Wat Jarvis aan Claude geeft en wat Claude antwoordt")
            st.warning(
                "Dit is een transparante veto-laag, geen chat die iedere marktcheck betaalt. "
                "Claude kan een voorstel alleen goedkeuren of blokkeren. Een setup die één "
                "keer is geweigerd wordt niet opnieuw gestuurd zolang hij niet wezenlijk "
                "verandert — zie het derde tabblad."
            )
            render_ai_exchange()
        with manage_view:
            st.subheader("Claude beheert de open posities")
            st.info(
                "Bij een openstaande positie mag Claude vijf dingen: vasthouden, de stop "
                "strakker zetten, het target dichterbij halen, deels sluiten of helemaal "
                "sluiten. Een stop ruimer zetten, het target verder weg leggen, bijkopen of "
                "omdraaien wordt geweigerd voordat het de broker bereikt."
            )
            render_supervision()
        with learn_view:
            st.subheader("Wat dit account zichzelf heeft geleerd")
            render_learning()

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
            st.plotly_chart(figure, width="stretch")

    with positions_tab:
        render_live_positions(account)
        st.divider()
        render_management_log()
        st.divider()
        render_trade_history()
        st.divider()
        st.subheader("Paper positions")
        if paper is None or not paper.positions:
            st.info("No simulated positions.")
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
                        for p in paper.positions
                    ]
                ),
                width="stretch",
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
            width="stretch",
        )

    with control_tab:
        control_notice = st.session_state.pop("control_notice", "")
        if control_notice:
            st.success(control_notice, icon=":material/check_circle:")
        control_error = st.session_state.pop("control_error", "")
        if control_error:
            st.error(control_error, icon=":material/error:")
        st.metric("Jarvis service", f"RUNNING (PID {running_pid})" if running else "OFF")
        with st.container(border=True):
            st.subheader("Claude trade gate")
            ai_state = "READY — FAIL CLOSED" if ai_ready else "BLOCKED"
            st.metric("Status", ai_state)
            st.caption(
                f"Provider: {settings.ai.provider} · model: "
                f"{settings.ai.anthropic_model or 'not configured'}"
            )
            latest_reviews = read_recent_reviews(ROOT / "runtime" / "ai_reviews.jsonl")
            if latest_reviews:
                latest = latest_reviews[-1]
                st.write(
                    {
                        "last_event": latest.get("event"),
                        "timestamp": latest.get("timestamp"),
                        "symbol": latest.get("symbol")
                        or (latest.get("outcome") or {}).get("symbol"),
                        "decision": latest.get("decision") or latest.get("reflection"),
                    }
                )
            else:
                st.info("No Claude trade review has been recorded yet.")
        if heartbeat_path.exists():
            st.json(heartbeat_path.read_text(encoding="utf-8"), expanded=False)
        stopped = kill_switch.is_engaged()
        st.metric("Hard STOP", "ENGAGED" if stopped else "clear")
        if stopped:
            st.error(
                "STOP is active. New entries are blocked; Jarvis closes its own positions "
                "and exits. This can take up to one scan interval (about 30 seconds). "
                f"Reason: {kill_switch.reason() or 'operator stop'}",
                icon=":material/stop_circle:",
            )
        st.warning(
            "Standard LIVE remains locked behind the validation protocol. EXPERIMENTAL LIVE "
            "is a separate real-money mode using the owner's explicit loss acceptance."
        )
        if experimental_contract is not None and not experimental_error:
            st.error(
                f"EXPERIMENTAL LIVE ARMED - account {experimental_contract.login}; "
                f"1.0% per trade; 15.0% drawdown stop; absolute equity floor "
                f"{experimental_contract.equity_floor:.2f} {experimental_contract.currency}."
            )
        else:
            st.info(f"Experimental live unavailable: {experimental_error}")
        stop_label = "STOP IS ENGAGED" if stopped else "STOP AND FLATTEN JARVIS"
        if st.button(
            stop_label,
            type="primary",
            width="stretch",
        ):
            if stopped:
                st.session_state["control_notice"] = "Hard STOP is already engaged."
            else:
                kill_switch.engage("operator dashboard emergency stop")
                st.session_state["control_notice"] = (
                    "Hard STOP engaged. Jarvis is blocking entries and flattening its own "
                    "positions."
                )
            st.rerun()
        start_monitor, start_paper, start_demo, start_experimental = st.columns(4)
        creation_flags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )

        def launch_mode(operation: str) -> None:
            subprocess.Popen(
                [sys.executable, str(ROOT / "jarvis.py"), "--operation", operation],
                cwd=ROOT,
                creationflags=creation_flags,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        def attempt_start(operation: str, blockers: list[str]) -> None:
            if blockers:
                st.session_state["control_error"] = "Cannot start: " + "; ".join(blockers)
            else:
                launch_mode(operation)
                st.session_state["control_notice"] = f"Jarvis {operation.upper()} started."
            st.rerun()

        common_start_blockers = []
        if stopped:
            common_start_blockers.append("hard STOP is active; use the reset form below")
        if running:
            common_start_blockers.append(f"Jarvis is already running as PID {running_pid}")

        if start_monitor.button("Start MONITOR", width="stretch"):
            attempt_start("monitor", [*common_start_blockers])
        if start_paper.button("Start PAPER", width="stretch"):
            attempt_start("paper", [*common_start_blockers])
        if start_demo.button(
            "Start DEMO",
            help=(
                "Log MT5 into a demo account first; the runner hard-refuses DEMO on live."
                if not account.is_demo
                else None
            ),
            width="stretch",
        ):
            demo_blockers = [*common_start_blockers]
            if not account.is_demo:
                demo_blockers.append("MT5 is logged into a live account, not demo")
            attempt_start("demo", demo_blockers)
        if start_experimental.button(
            "Start EXPERIMENTAL LIVE",
            help=(
                (
                    "Claude gate is not ready; live starts fail closed."
                    if not ai_ready
                    else experimental_error
                )
                if (not ai_ready or experimental_error)
                else "Starts autonomous orders with real money on the bound account."
            ),
            type="primary",
            width="stretch",
        ):
            experimental_blockers = [*common_start_blockers]
            if account.is_demo:
                experimental_blockers.append("MT5 is logged into a demo account")
            if experimental_contract is None or experimental_error:
                experimental_blockers.append(
                    experimental_error or "experimental contract is unavailable"
                )
            if not ai_ready:
                experimental_blockers.append("Claude API gate is not ready")
            attempt_start("experimental_live", experimental_blockers)
        with st.form("stop_reset_and_start", border=True):
            st.subheader("STOP resetten of Jarvis opnieuw starten")
            confirmation = st.text_input(
                "Type clear stop to confirm",
                placeholder="clear stop",
                help="Capitalization and extra spaces do not matter.",
                key="clear_stop_confirmation",
            )
            st.caption(
                "De knoppen zijn altijd klikbaar. Na de klik krijg je direct een exacte "
                "foutmelding als een veiligheidsvoorwaarde niet klaar is."
            )
            clear_only, clear_and_live = st.columns(2)
            clear_only_submitted = clear_only.form_submit_button(
                "Clear STOP only",
                type="secondary",
                width="stretch",
            )
            clear_and_live_submitted = clear_and_live.form_submit_button(
                "CLEAR STOP + START REAL TRADING",
                type="primary",
                help="Clears STOP and immediately starts account-bound Experimental Live.",
                width="stretch",
            )

        confirmation_ok = " ".join(confirmation.split()).casefold() == "clear stop"
        if clear_only_submitted:
            if not confirmation_ok:
                st.session_state["control_error"] = "Type exact: clear stop"
            elif not stopped:
                st.session_state["control_notice"] = "Hard STOP was already clear."
            else:
                kill_switch.clear()
                st.session_state["control_notice"] = (
                    "Hard STOP cleared. Jarvis remains off until you explicitly start a mode."
                )
            st.rerun()

        if clear_and_live_submitted:
            blockers = []
            if not confirmation_ok:
                blockers.append("type exact: clear stop")
            if running:
                blockers.append(f"Jarvis is already running as PID {running_pid}")
            if account.is_demo:
                blockers.append("MT5 is logged into a demo account")
            if experimental_contract is None or experimental_error:
                blockers.append(experimental_error or "experimental contract is unavailable")
            if not ai_ready:
                blockers.append("Claude API gate is not ready")
            if blockers:
                st.session_state["control_error"] = "Cannot start: " + "; ".join(blockers)
            else:
                kill_switch.clear()
                subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "jarvis.py"),
                        "--operation",
                        "experimental_live",
                    ],
                    cwd=ROOT,
                    creationflags=creation_flags,
                    close_fds=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                st.session_state["control_notice"] = (
                    "STOP cleared and EXPERIMENTAL LIVE started with real money."
                )
            st.rerun()
finally:
    service.close()
