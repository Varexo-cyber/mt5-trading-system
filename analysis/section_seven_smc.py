"""Stateful, replayable gold liquidity-sweep reversal."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from config.schema import SectionSevenSmcConfig
from core.types import MarketContext, Signal, Timeframe

def _atr(f, period=14):
    prev=f["close"].shift(1)
    tr=pd.concat([f["high"]-f["low"],(f["high"]-prev).abs(),(f["low"]-prev).abs()],axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

@dataclass
class _Setup:
    direction:int; swept_at:object; extreme:float; liquidity:float; structure:float
    stage:str="displacement"; gap_low:float|None=None; gap_high:float|None=None; displaced_at:object|None=None

class SectionSevenGoldSmc:
    """Track sweep -> BOS/CHOCH displacement -> FVG -> defended retest."""
    name="section_seven_gold_smc"
    def __init__(self,config=None):
        self.config=config or SectionSevenSmcConfig(); self._last_trade_day={}; self._last_seen_bar={}; self._setups={}

    @staticmethod
    def _pivots(s,span,is_high):
        a=s.astype(float).tolist(); out=[]
        for i in range(span,len(a)-span):
            w=a[i-span:i+span+1]
            if (is_high and a[i]==max(w)) or (not is_high and a[i]==min(w)): out.append(a[i])
        return out

    @staticmethod
    def _age(f,stamp):
        pos=f.index.get_indexer([stamp])[0]
        return len(f) if pos<0 else len(f)-1-int(pos)

    def _new_sweep(self,f,unit):
        c=self.config; hist=f.iloc[-(c.liquidity_lookback+1):-1]
        ph=self._pivots(hist.high,c.pivot_span,True); pl=self._pivots(hist.low,c.pivot_span,False)
        if not ph or not pl:return None
        h,l,x=map(float,(f.high.iloc[-1],f.low.iloc[-1],f.close.iloc[-1]))
        loc=(x-l)/max(h-l,unit*.01); excursion=c.sweep_excursion_atr*unit
        sh=h>=ph[-1]+excursion and x<ph[-1] and loc<=c.sweep_close_location
        sl=l<=pl[-1]-excursion and x>pl[-1] and loc>=1-c.sweep_close_location
        if sh==sl:return None
        d=-1 if sh else 1
        return _Setup(d,f.index[-1],h if d<0 else l,ph[-1] if d<0 else pl[-1],pl[-1] if d<0 else ph[-1])

    def _advance(self,s,f,unit):
        c=self.config; d=s.direction; age=self._age(f,s.swept_at); x=float(f.close.iloc[-1])
        if (d>0 and float(f.low.iloc[-1])<s.extreme) or (d<0 and float(f.high.iloc[-1])>s.extreme):return "invalid"
        if s.stage=="displacement":
            if age==0:return "waiting"
            if age>c.displacement_window_bars:return "invalid"
            broke=x>s.structure if d>0 else x<s.structure
            body=abs(float(f.close.iloc[-1]-f.open.iloc[-1]))
            if not broke or body<c.displacement_body_atr*unit:return "waiting"
            lo,hi=(float(f.high.iloc[-3]),float(f.low.iloc[-1])) if d>0 else (float(f.high.iloc[-1]),float(f.low.iloc[-3]))
            if hi-lo<c.fvg_minimum_atr*unit:return "waiting"
            s.stage="retest";s.gap_low=lo;s.gap_high=hi;s.displaced_at=f.index[-1];return "waiting"
        age=self._age(f,s.displaced_at)
        if age==0:return "waiting"
        if age>c.retest_window_bars:return "invalid"
        tol=c.retest_tolerance_atr*unit
        touched=float(f.low.iloc[-1])<=s.gap_high+tol and float(f.high.iloc[-1])>=s.gap_low-tol
        held=x>=s.gap_low if d>0 else x<=s.gap_high
        return "ready" if touched and held else "waiting"

    def analyze(self,ctx:MarketContext)->Signal:
        c=self.config
        if not c.enabled or ctx.symbol not in c.allowed_symbols:return Signal.neutral(self.name,"section seven SMC disabled for this market")
        clock=Timeframe.parse(c.timeframe); series=ctx.series.get(clock); required=(Timeframe.M15,Timeframe.M30,Timeframe.H1,Timeframe.H4)
        if series is None or any(ctx.series.get(t) is None for t in required):return Signal.neutral(self.name,"S7 needs entry, M15, M30, H1 and H4 closed bars")
        f=series.df
        if len(f)<c.liquidity_lookback+c.pivot_span+8:return Signal.neutral(self.name,"S7 needs confirmed swing structure")
        key=f"{ctx.symbol}:{c.timeframe}";stamp=f.index[-1]
        if self._last_seen_bar.get(key)==stamp:return Signal.neutral(self.name,"S7 already assessed this closed bar")
        self._last_seen_bar[key]=stamp;today=stamp.date()
        if self._last_trade_day.get(key)==today:return Signal.neutral(self.name,"S7 already produced its one setup for this UTC day")
        unit=_atr(f.iloc[:-1])
        if not pd.notna(unit) or unit<=0:return Signal.neutral(self.name,"S7 ATR is unavailable")
        s=self._setups.get(key)
        if s is None:
            s=self._new_sweep(f,unit)
            if s is None:return Signal.neutral(self.name,"no confirmed external liquidity sweep")
            self._setups[key]=s;return Signal.neutral(self.name,"liquidity swept; awaiting displacement/BOS")
        state=self._advance(s,f,unit)
        if state=="invalid":self._setups.pop(key,None);return Signal.neutral(self.name,"SMC sequence expired or invalidated")
        if state!="ready":return Signal.neutral(self.name,f"SMC sequence awaiting {s.stage}")
        d=s.direction;h1=ctx.series[Timeframe.H1].df;hh=float(h1.high.iloc[-c.context_lookback:].max());ll=float(h1.low.iloc[-c.context_lookback:].min());mid=(hh+ll)/2
        entry=ctx.tick.mid if ctx.tick is not None else float(f.close.iloc[-1])
        if (d<0 and entry<mid) or (d>0 and entry>mid):self._setups.pop(key,None);return Signal.neutral(self.name,"outside H1 premium/discount")
        biases=[];slopes=[]
        for tf in required:
            z=ctx.series[tf].df.close.astype(float).iloc[-c.context_lookback:];half=max(2,len(z)//2);slope=float(z.iloc[-half:].median()-z.iloc[:half].median());slopes.append(slope);biases.append(1 if slope>0 else -1 if slope<0 else 0)
        if biases[-2]==biases[-1]==-d:self._setups.pop(key,None);return Signal.neutral(self.name,"H1 and H4 oppose reversal")
        stop=s.extreme-d*c.stop_buffer_atr*unit;risk=abs(entry-stop);target=hh if d>0 else ll;rr=d*(target-entry)/risk if risk else -1
        if d*(entry-stop)<=0 or rr<c.minimum_liquidity_reward_r:self._setups.pop(key,None);return Signal.neutral(self.name,"opposing liquidity offers insufficient R")
        self._setups.pop(key,None);self._last_trade_day[key]=today
        return Signal(module=self.name,score=c.score*d,confidence=c.confidence,reasoning="liquidity sweep, displacement/BOS, FVG retest and HTF structure",invalidation_price=stop,key_levels=(s.liquidity,s.gap_low,s.gap_high,mid,target),details={"timeframe":c.timeframe,"htf_biases":biases,"trend_slopes":slopes,"liquidity_target":target,"liquidity_reward_r":rr})
