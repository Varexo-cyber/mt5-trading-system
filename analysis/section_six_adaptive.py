"""Section six: frozen monthly walk-forward models for gold and SPX500."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.schema import SectionSixModelConfig
from core.types import MarketContext, Signal, Timeframe

_RNG = np.random.default_rng(7401)
_PROJECTION = _RNG.normal(0.0, 0.55, size=(13, 48))
_OFFSET = _RNG.uniform(-1.0, 1.0, size=48)


@dataclass(frozen=True, slots=True)
class FrozenModel:
    beta: tuple[float, ...]
    centre: tuple[float, ...]
    scale: tuple[float, ...]


_GOLD = FrozenModel(
    beta=(
        -0.16493640869716503,
        0.09272327605191544,
        -0.0869972664233162,
        -0.05154717757616192,
        -0.00726794314437223,
        -0.04404167949737471,
        0.008366089647840717,
        -0.007159143404042482,
        -0.05123784425421671,
        -0.14037702414431466,
        -0.018097438634931115,
        0.03896547432049118,
        -0.0931125812217817,
        -0.1328187385921253,
        -0.12222310160459232,
        0.035395254768577744,
        0.1223670381070187,
        -0.019904203873135663,
        0.009762449191057954,
        -0.04986883539682293,
        -0.017862381554922046,
        -0.11036844848833573,
        0.07401070369018409,
        0.04956912543129706,
        0.05173140502751518,
        -0.06070638438891588,
        0.11162017681452938,
        -0.08544076554442526,
        -0.003965795972814421,
        -0.11293929841163912,
        -0.024562547502812875,
        0.06241744445994536,
        0.07601522663884978,
        -0.03565379139472855,
        0.11617768843457077,
        -0.06335870965971076,
        0.016055578915743312,
        -0.0786584543243085,
        0.04421259620844426,
        -0.008827491917254194,
        -0.00017434269208171945,
        0.028880337135896834,
        0.023416289953728697,
        -0.08800153288730225,
        0.02023214653996515,
        -0.0035534867073363824,
        -0.04367057139999869,
        0.007687747722874565,
        0.10921122596693235,
    ),
    centre=(
        -0.012432197765029716,
        -0.03715775285109561,
        -0.07252506830288406,
        -0.13266950099950992,
        -0.10261299340529025,
        -0.04085235922555382,
        -0.014189435903031816,
        1.0059359824284946,
        -0.00757995483848382,
        0.025055615628366903,
        0.04227659251086276,
        0.03407607681585875,
        -0.03577087935267166,
    ),
    scale=(
        0.6845280639713355,
        1.1464251294384673,
        1.569404952066652,
        2.1501222516383653,
        1.2008831510306115,
        0.8052458206502501,
        0.6769358652905063,
        0.47521909256824507,
        0.31110483185727733,
        0.3610357016643326,
        0.39937152810790505,
        0.7078430716728825,
        0.7046399442721597,
    ),
)

_SPX = FrozenModel(
    beta=(
        0.20182253843153716,
        0.0250964443124818,
        -0.008232505853711899,
        0.22728191464845737,
        -0.05702039690235712,
        -0.2018335613792985,
        0.13680879362578421,
        0.16199143620712278,
        0.19129169890600373,
        -0.198422492605586,
        0.2269823548964841,
        0.06944024617537962,
        0.036929697899795604,
        0.19777670170022096,
        0.0008036098674864613,
        0.2094805378182465,
        0.10784318824847136,
        0.10269990487765744,
        -0.10544075106783013,
        -0.06266628186837221,
        -0.016633273604681274,
        -0.11959828152206825,
        -0.2048962139905261,
        -0.0032971133273143466,
        -0.22112275829939598,
        -0.04279010825372389,
        -0.060956069705341974,
        0.035782939541825154,
        -0.06959197206563646,
        0.2202012939517959,
        0.10533074595825787,
        -0.06193398005398347,
        -0.10760159826847392,
        -0.07489122393163467,
        -0.1320281672373409,
        -0.09629310660328899,
        -0.0942410247532225,
        0.05087940174497983,
        0.25633124325945505,
        0.14796253136650991,
        0.18728613137575129,
        0.01533186644872688,
        -0.1264768714058897,
        0.1206702601110415,
        0.6154896541857252,
        0.11964191318903244,
        0.28731244234785136,
        -0.0906501746510936,
        -0.09043255544229442,
    ),
    centre=(
        -0.017369500508559513,
        -0.03389485839593384,
        -0.03968360510421267,
        0.00261982141320108,
        0.06754593291609035,
        -0.018586613221008845,
        -0.022603794243218924,
        1.0374764967278627,
        0.014903536806480726,
        0.029329652753436984,
        0.12140543462012225,
        0.03873735545743797,
        -0.03201385483284561,
    ),
    scale=(
        0.7241750947194737,
        1.1890744522715913,
        1.5529358777305367,
        2.100563022169905,
        1.1987927015230537,
        0.8122049346305393,
        0.7146912297904096,
        0.6486157023861351,
        0.30726267722676764,
        0.31986406888671565,
        0.5291895653249732,
        0.703707871315157,
        0.7087099281370289,
    ),
)


def _atr(frame: pd.DataFrame) -> pd.Series:
    previous = frame["close"].shift(1)
    return (
        pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(14)
        .mean()
    )


def model_reading(frame: pd.DataFrame, model: FrozenModel) -> tuple[float, float] | None:
    if len(frame) < 80:
        return None
    close, open_ = frame["close"].astype(float), frame["open"].astype(float)
    high, low = frame["high"].astype(float), frame["low"].astype(float)
    unit = _atr(frame).replace(0.0, np.nan)
    span = (high - low).replace(0.0, np.nan)
    fast = close.ewm(span=8, adjust=False).mean()
    slow = close.ewm(span=32, adjust=False).mean()
    volume = frame.get("volume", frame.get("tick_volume", pd.Series(0.0, index=frame.index)))
    volume = volume.astype(float)
    hour = frame.index[-1].hour + frame.index[-1].minute / 60.0
    values = np.asarray(
        [
            close.diff(1).iloc[-1] / unit.iloc[-1],
            close.diff(3).iloc[-1] / unit.iloc[-1],
            close.diff(6).iloc[-1] / unit.iloc[-1],
            close.diff(12).iloc[-1] / unit.iloc[-1],
            (fast.iloc[-1] - slow.iloc[-1]) / unit.iloc[-1],
            (close.iloc[-1] - fast.iloc[-1]) / unit.iloc[-1],
            (close.iloc[-1] - open_.iloc[-1]) / unit.iloc[-1],
            span.iloc[-1] / unit.iloc[-1],
            (close.iloc[-1] - low.iloc[-1]) / span.iloc[-1] - 0.5,
            unit.iloc[-1] / unit.rolling(48).mean().iloc[-1] - 1.0,
            volume.iloc[-1] / volume.rolling(48).median().replace(0.0, np.nan).iloc[-1] - 1.0,
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
        ]
    )
    if not np.isfinite(values).all():
        return None
    centre = np.asarray(model.centre)
    scale = np.asarray(model.scale)
    hidden = np.tanh(((values - centre) / scale) @ _PROJECTION + _OFFSET)
    beta = np.asarray(model.beta)
    return float(beta[0] + hidden @ beta[1:]), float(unit.iloc[-1])


class _SectionSixModel:
    name = ""
    symbol = ""
    model = _GOLD

    def __init__(self, config: SectionSixModelConfig | None = None) -> None:
        self.config = config or SectionSixModelConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        if not cfg.enabled or ctx.symbol != self.symbol:
            return Signal.neutral(self.name, "section six disabled for this market")
        timeframe = Timeframe.parse(cfg.timeframe)
        series = ctx.series.get(timeframe)
        found = model_reading(series.df, self.model) if series is not None else None
        if found is None:
            return Signal.neutral(self.name, f"section six needs 80 closed {cfg.timeframe} bars")
        reading, unit = found
        directed = reading * cfg.polarity
        if abs(directed) < cfg.threshold:
            return Signal.neutral(self.name, f"model magnitude {abs(directed):.3f} below threshold")
        direction = 1 if directed > 0.0 else -1
        if cfg.long_only and direction < 0:
            return Signal.neutral(self.name, "section six gold route is long-only")
        if cfg.session_start_hour_utc is not None and cfg.session_end_hour_utc is not None:
            hour = ctx.now.hour
            start, end = cfg.session_start_hour_utc, cfg.session_end_hour_utc
            inside = start <= hour < end if start < end else hour >= start or hour < end
            if not inside:
                return Signal.neutral(
                    self.name,
                    f"outside measured {start:02d}:00-{end:02d}:00 UTC gold session",
                )
        # ONE HOUR OF MOMENTUM IN THE SAME DIRECTION, on already-closed bars.
        #
        # The full 180-day broker replay put this route at 885 trades, 24.5%
        # and -71.65R. The strong recent month was regime, not edge. This is
        # the one causal filter that survived being split before it was
        # looked at: first 90 days +57.16R, next 45 validated +4.14R, and only
        # then the newest 45 at +46.33R.
        #
        # Causal on purpose -- `confirmation_bars` CLOSED bars back, no part of
        # the forming bar -- because a filter that peeks is how a backtest
        # invents an edge that live cannot take.
        #
        # It is a filter, not a rescue. June and July stayed negative in the
        # exact engine replay and the drawdown is still around -51R. The owner
        # is forward-testing this knowingly.
        if cfg.confirmation_bars > 0:
            closes = series.df["close"].astype(float)
            if len(closes) <= cfg.confirmation_bars:
                return Signal.neutral(
                    self.name, f"needs {cfg.confirmation_bars + 1} closed {cfg.timeframe} bars"
                )
            drift = float(closes.iloc[-1]) - float(closes.iloc[-1 - cfg.confirmation_bars])
            if drift * direction <= 0.0:
                return Signal.neutral(
                    self.name,
                    f"{cfg.confirmation_bars} closed {cfg.timeframe} bars do not confirm "
                    f"the direction ({drift:+.2f})",
                )

        price = ctx.tick.mid if ctx.tick is not None else float(series.df["close"].iloc[-1])
        return Signal(
            module=self.name,
            score=cfg.score * direction,
            confidence=cfg.confidence,
            reasoning=f"monthly walk-forward {cfg.timeframe} reading {directed:+.3f}",
            invalidation_price=price - direction * cfg.stop_atr * unit,
            details={"timeframe": cfg.timeframe, "model_reading": round(directed, 6)},
        )


class SectionSixGoldM5(_SectionSixModel):
    name = "section_six_gold_m5"
    symbol = "XAUUSD"
    model = _GOLD


class SectionSixSpxH1(_SectionSixModel):
    name = "section_six_spx_h1"
    symbol = "SPX500"
    model = _SPX
