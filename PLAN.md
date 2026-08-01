# Plan — autonoom MT5 trading systeem

Status: **Fase 1, 2 en 3 af.** Alles hieronder is de afspraak voor de rest.

Documenten zijn in het Nederlands (die lees jij), code en docstrings in het
Engels (dat is de taal van de libraries waar het tussen staat, en gemengde
codebases lezen slecht). Als je dat liever anders hebt, zeg het nu — later
omzetten is duur.

---

## 0. Eerst het vervelende nieuws

Voordat we verder bouwen moet dit op tafel, want het bepaalt wat de rest van
het project realistisch kan opleveren.

### 0.1 €100 is te weinig om jouw eigen risicoregels te volgen

De rekensom, met jouw eigen getallen:

| | EURUSD | USDJPY | XAUUSD |
|---|---|---|---|
| minimum lot | 0.01 | 0.01 | 0.01 |
| waarde per pip bij min. lot | ≈ €0,92 | ≈ €0,62 | ≈ €0,92 per $0,01 |
| SL die 1% (€1) kost | **≈ 11 pips** | ≈ 16 pips | ≈ 1 dollar koersbeweging |
| SL die 2% (€2) kost | **≈ 22 pips** | ≈ 32 pips | ≈ 2 dollar koersbeweging |
| kosten van een normale 30-pip SL | 3% | 2% | n.v.t. |

Een structurele stoploss — achter een swing low, achter een order block, plus
een ATR-buffer tegen spread-hunting — is op EURUSD H1 zelden onder de 20 pips
en vaak 30–50. Op €100 kun je die dus niet betalen zonder je eigen 1%-regel te
breken.

Dat is geen bug en geen strategieprobleem. Het is rekenkunde. De drie eerlijke
uitwegen:

1. **Accepteer dat het systeem bijna alles skipt** met reden
   `TRADE_SKIPPED_UNDERCAPITALIZED`. Prima voor Fase 1–7, want daar valideren
   we code, niet winst. Maar het levert geen live trades op in Fase 8.
2. **Zoek een broker met 0.001 lots** (micro-/cent-accounts bestaan). Dan is
   1% risico bij een 30-pip SL wél uitdrukbaar, en wordt Fase 8 zinvol.
   Dit is wat ik zou doen.
3. **Fund de rekening met €500–1000** voor Fase 8. Bij €500 koopt 1% een SL
   van ~54 pips op EURUSD, en dan is er ruimte voor echte setups.

Wat we **niet** doen: het risico opschroeven zodat de trades passen, of naar de
minimum lot afronden. Dat tweede is de klassieke doodsoorzaak van kleine
accounts — de trade die 1% zou riskeren riskeert stilletjes 4%, en vijf daarvan
op rij is de helft van je geld. De `PositionSizer` in Fase 2 skipt in plaats van
af te ronden, en `core/startup.py` rekent bij elke start voor wat je account
wél aankan (`--status` laat het zien).

Ik heb `max_sl_pips: 30` in `micro_live` laten staan zoals je vroeg. Merk op dat
het risicoplafond van 2% bij €100 al bij ~22 pips bindt, dus die 30 wordt in de
praktijk nooit de bindende beperking. Beide checks staan er en beide loggen hun
eigen reden.

### 0.2 De €100 in Fase 8 is statistisch leeg — en dat is prima

Je zei het zelf al goed: het slagingscriterium van Fase 8 is nul onverklaarde
verschillen tussen wat het systeem dacht te doen en wat MT5 deed. Ik bouw
`EXECUTION_REPORT.md` daaromheen en zet bij elk P&L-getal expliciet het
betrouwbaarheidsinterval, zodat er niet op ruis bijgestuurd wordt.

### 0.3 De kennisbasis is te groot voor één systeem

Secties A t/m O in je opdracht zijn samen makkelijk 60+ modules. Met ~100 trades
per module als minimum steekproef, en 2–3 trades per week, is dat decennia aan
validatie. Overfitting is dan niet een risico maar een zekerheid.

Mijn voorstel: bouw ze allemaal als *modules* (ze zijn goedkoop en los
testbaar), maar geef alleen gewicht aan een kleine kern die zich bewezen heeft.
Concreet voor Fase 4–5, in deze volgorde:

1. Market structure (BOS/CHoCH, swing points) — hier komt je SL-plaatsing uit
2. Levels (S/R, dag/week-open, ronde getallen)
3. Liquidity sweeps + order blocks (de twee SMC-concepten met de duidelijkste
   economische logica)
4. Volatiliteitsregime (ATR/ADX) — geen signaal, wel de schakelaar tussen
   trending- en range-setups
5. Sessie/timing

Dat is vijf modules. Elliott Wave, Wyckoff-fasen, seizoenspatronen en COT komen
later of nooit — ze staan in de architectuur, met gewicht 0, tot een backtest
iets anders zegt. Ik zeg dit vooraf omdat het achteraf altijd voelt alsof je
iets weggooit waar werk in zit.

---

## 1. Wat er nu staat (Fase 1 t/m 3)

```
mt5-trading-system/
├── config/
│   ├── config.yaml          alle parameters, geen magic numbers in code
│   ├── schema.py            pydantic-modellen; de harde regels zijn hier validaties
│   ├── loader.py            YAML + overlay + TS_-env, met leesbare foutmeldingen
│   └── .env.example         credentials-template (.env staat in .gitignore)
├── core/
│   ├── types.py             Signal, MarketContext, OrderRequest/Result, Position
│   ├── errors.py            exceptionhiërarchie
│   ├── clock.py             LiveClock / SimulatedClock — niets roept datetime.now()
│   ├── mt5_codes.py         MT5-constanten, gespiegeld zodat het op Linux importeert
│   ├── instrument.py        pip-wiskunde, lot-afronding, risicohaalbaarheid
│   ├── mt5_connector.py     verbinding, reconnect, order execution, retries
│   ├── data_manager.py      OHLCV, multi-timeframe, caching, datakwaliteit
│   └── startup.py           startup-guard + haalbaarheidsrapport
├── filters/
│   ├── base.py              Filter-protocol + FilterChain (stopt bij eerste block)
│   ├── news_filter.py       VERPLICHT, fail-closed
│   ├── calendar/            events, providers (2 remote + file), fail-safe service
│   ├── session_filter.py    Londen/NY/Azië, rollover, weekendranden
│   ├── spread_filter.py     zelflerende baseline per instrument per uur
│   └── correlation_filter.py  rollende correlatie, richtingsbewust
├── risk/
│   ├── reasons.py           gesloten vocabulaire van redenen (gaat in de journal)
│   ├── position_sizer.py    lotberekening + de undercapitalized-check
│   └── risk_manager.py      alle limieten, circuit breaker, verboden praktijken
├── journal/
│   ├── database.py          SQLite-schema, migraties, risk-queries
│   └── recorder.py          schrijft cycli, trades, executies, shadow trades
├── infra/
│   ├── logging.py           JSON naar bestand, leesbaar naar console
│   └── killswitch.py        het STOP-bestand
├── scripts/verify_calendar.py     verifieert de kalenderbronnen live
├── tests/                   282 tests, groen, draaien zonder MT5
├── scripts/phase1_acceptance.py   de demo-order test (draai dit op Windows)
└── main.py                  --check-config / --status / --risk / --filters / --data
```

### Keuzes die ik gemaakt heb, en waarom

**De hardware regels zijn validaties, geen documentatie.** Martingale, grid,
averaging down, handelen zonder stoploss en een nieuwsfilter dat "fail open"
gaat kun je niet aanzetten door YAML te bewijzen — de config weigert te laden.
Om ze aan te zetten moet je `config/schema.py` aanpassen, en dat is een
zichtbare, reviewbare handeling in plaats van een typefout op een slechte dag.
Zie de tests in `tests/test_config.py::TestHardRules`.

**De vormende candle wordt nooit teruggegeven.** `DataManager` gooit elke bar
weg die nog niet gesloten is, op basis van de klok en niet op basis van "gooi de
laatste rij weg". Dit is de belangrijkste eigenschap van de hele datalaag: zonder
die garantie scoort een indicator een setup op een prijs die nog niet gebeurd is,
en dat is precies waarom backtests mooi zijn en live accounts leeglopen.

**Niets roept `datetime.now()` aan.** Alles vraagt een `Clock`. Dat is wat het
straks mogelijk maakt om het nieuwsfilter en de time-based exit in een backtest
via exact hetzelfde codepad te replayen.

**Lotsize wordt altijd naar beneden afgerond.** Nooit naar dichtstbijzijnde.
Naar boven afronden verhoogt het risico boven wat de sizer berekende, en maakt
je risicopercentage een leugen.

**Verliezen worden op equity gemeten, niet op gerealiseerde P&L.** Een open
verliezer telt dus meteen mee voor de daglimiet. Anders kan de rekening 8%
onder water staan met een "0% dagverlies" en vrolijk nieuwe posities openen.

**De dag rolt om 21:00 UTC, niet om middernacht.** Dat is de FX-rollover.
Bij middernacht UTC zou de daglimiet midden in de New York-sessie resetten.
De week begint op de zondag-grens, want de FX-week opent zondagavond.

**De equity-ankers staan in de journal, niet in het geheugen.** Een herstart om
14:00 na 2% verlies mag geen vers dagbudget teruggeven. `INSERT OR IGNORE` op
de dag/week-grens is precies wat dat voorkomt.

**Slippage heeft een richtingsbewust teken.** Positief = slechter dan gevraagd,
of je nu long of short zat. Ruwe prijsverschillen middelen over longs en shorts
heft echte slippage bijna precies op tot nul, en dan denk je dat je executie
perfect is.

**Alleen tijdelijke rejecties worden geretried.** Requote, timeout,
price-changed: opnieuw proberen. `NO_MONEY`, `INVALID_STOPS`, `MARKET_CLOSED`:
één poging, loggen, klaar. Een retry-loop op `NO_MONEY` is hoe een account
langzaam wordt uitgeknepen.

**De MT5-constanten staan gespiegeld in `core/mt5_codes.py`** zodat de hele
codebase op Linux importeert en de testsuite overal draait. Bij `connect()`
wordt gecontroleerd of ze nog overeenkomen met het geïnstalleerde pakket — als
een terminal-update ooit een timeframe hernummert, wordt dat een luide
startfout in plaats van analyse op de verkeerde timeframe.

**`InstrumentSpec` gebruikt de tick_value van de broker**, die in
*accountvaluta* staat. Daarmee is de valutaconversie voor een EUR-rekening die
USD-paren handelt het probleem van de broker en niet van ons — een eigen
conversietabel is een extra plek waar iets stil fout kan gaan.

**Verboden praktijken crashen, ze worden niet geweigerd.** Een gate die "nee"
zegt is normaal en wordt gelogd. Maar averaging down, hedgen van hetzelfde
symbool, of het risico verhogen na een verlies gooien een `ForbiddenStrategyError`.
Als een strategie dat probeert is er iets fundamenteel mis, en doorgaan zou het
verbod adviserend maken.

**Geen kalender betekent geen trade.** Het nieuwsfilter blokkeert als beide
bronnen down zijn én de cache verlopen is. Een lege kalender en een ontbrekende
kalender zijn van buitenaf niet te onderscheiden, en één keer verkeerd gokken
tijdens een NFP kost meer dan alle setups die het filter ooit zal skippen.
De cache verloopt (`max_calendar_age_minutes`, standaard 3 uur) — een kalender
van vier uur oud weet niets van een verplaatste of toegevoegde release.

**Een parser die 3 van de 40 events teruggeeft is gevaarlijker dan één die
crasht.** De ontbrekende 37 zijn onzichtbaar en het filter meldt "veilig" voor
een venster dat het had moeten blokkeren. Elke parser telt daarom hoeveel
records hij niet kon lezen en laat de hele fetch falen boven 10%.

**Onbekende correlatie blokkeert.** Als de correlatie tussen een kandidaat en
een open positie niet te meten is (te weinig overlappende bars, ontbrekende
data), is dat geen bewijs dat ze onafhankelijk zijn.

### Eén ontwerpprobleem dat ik onderweg vond

`max_sl_pips` is een FX-regel. Een "pip" op goud is één point ($0,01), dus een
plafond van 30 pips zou daar een stoploss van 30 cent betekenen. Ik pas het
plafond daarom alleen op FX toe; voor de rest is de geldgebaseerde
undercapitalized-check de bindende grens, en die is exact in plaats van een
benadering. Als goud in fase 5 echt verhandelbaar wordt, wil je dit waarschijnlijk
in ATR-veelvouden uitdrukken in plaats van in pips. Genoteerd voor dan.

### Wat er nog niet is

Geen strategie, geen filters, geen orderloop. Dat is de bedoelde volgorde: het
vangnet (Fase 2) en het nieuwsfilter (Fase 3) komen vóór iets dat zelfstandig
orders kan versturen.

---

## 2. Zo controleer je Fase 1 zelf

Op elk platform (geen MT5 nodig):

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest              # 282 tests
.venv/bin/python main.py --check-config
```

Op je Windows-machine, met MT5 open en ingelogd op een **demo**-account:

```bash
copy config\.env.example config\.env    # en invullen
python main.py --status                 # startup-guard + haalbaarheidsrapport
python main.py --data EURUSD            # multi-timeframe overzicht + ATR
python scripts\phase1_acceptance.py --symbol EURUSD
```

Dat laatste script is de eigenlijke acceptatietest van Fase 1: het plaatst één
minimum-lot order met SL en TP, wacht, controleert of de broker onze stops heeft
overgenomen, sluit hem, en print requested vs. filled, slippage, latency en alle
returncodes. Het weigert te draaien op een live account.

**Wat ik van je terug wil zien:** de output van `--status` en van het
acceptance-script. Daarin staat je echte pip-waarde, je echte minimum lot en je
echte spread, en die drie getallen bepalen of Fase 8 zinvol is.

---

## 3. De rest van de fases

| Fase | Inhoud | Klaar wanneer |
|---|---|---|
| ~~2~~ | ~~Position sizer, risk manager, SQLite journal~~ | **Af** — 80 tests op sizing, limieten en journal |
| ~~3~~ | ~~Nieuwsfilter, sessie, spread, correlatie~~ | **Af** — 74 tests. Nog te doen: `verify_calendar.py` één keer draaien tegen de echte feeds |
| **4** | Analyse: market structure → levels → SMC → indicatoren. Eén module per keer, elk met een plot | Jij bevestigt op een chart dat de module ziet wat jij ziet |
| **5** | Confluence engine + backtester (pessimistisch: echte spreads, slippage, commissie, SL-eerst bij intrabar-ambiguïteit) | Walk-forward draait; look-ahead-test slaagt |
| **6** | Trade management + live loop op demo | Loop draait een week zonder handmatige interventie; reconciliatie clean |
| **7** | Monitoring, rapportage, postmortem | Wekelijks rapport genereert zichzelf |
| **8** | Micro-live shakeout | Nul onverklaarde discrepanties |
| **9** | Evaluatie & opschalen | `sample_size_check` zegt dat er genoeg trades zijn |

Fase 4 is het grootste blok en wordt per module opgeleverd.

---

## 4. Openstaande vragen

Deze houden Fase 2 niet tegen — ik begin daar gewoon aan. Maar ze bepalen wel
hoe Fase 3 en 8 eruitzien, dus hoe eerder hoe beter.

1. **Welke broker en welk account?** Ik heb de exacte symboolnamen nodig
   (`EURUSD`, `EURUSD.pro`, `EURUSDm`?), of het hedging of netting is, en of ze
   0.001 lots aanbieden. De output van `python main.py --status` beantwoordt dit
   allemaal in één keer.

2. **De accountgrootte-beslissing uit §0.1.** Blijft het €100 (en accepteren we
   dat Fase 8 vooral skips oplevert), gaan we naar een broker met kleinere lots,
   of wordt de rekening voor Fase 8 opgehoogd? Dit is de belangrijkste vraag in
   dit document.

3. ~~**Welke kalenderbron?**~~ Gekozen: ForexFactory via FairEconomy (primair)
   en TradingView's publieke endpoint (fallback), plus een lokale
   file-provider voor de backtest en als handmatige noodklep. **Maar:** deze
   omgeving blokkeert uitgaand HTTPS naar externe hosts, dus ik heb de parsers
   niet tegen echte responses kunnen draaien. Draai
   `python scripts/verify_calendar.py --raw` één keer op jouw machine en stuur
   me de output — dan corrigeer ik de parsers met echte data. **Doe dit vóór
   fase 8.** Waar je op let staat in de docstring van dat script; de
   belangrijkste: nul high-impact events in een normale week is de signatuur
   van een parser die het impact-veld kwijt is.

4. **Draait de PC 24/5 of wordt het een VPS?** Bij een PC die 's nachts uitgaat
   moet het systeem weten dat het posities kan aantreffen die het niet zelf
   beheerd heeft, en dat is andere reconciliatielogica.

5. **Alerts: Telegram of Discord?** Nodig voor de circuit breaker in Fase 2 —
   die moet je kunnen bereiken.

6. **Netting of hedging account?** Bij een netting-account werkt "twee posities
   tegelijk" fundamenteel anders (ze salderen), en dat raakt de risk manager in
   Fase 2 direct.

---

## 5. Wat ik niet ga bouwen, tenzij je erop staat

- **Een neuraal netwerk dat live zijn gewichten aanpast.** Je vroeg er niet om
  en schreef zelf al waarom niet. Volledig eens: de drie leerlagen uit je
  opdracht zijn de juiste aanpak.
- **Volledige Kelly.** ¼ Kelly staat in de config met een harde bovengrens van
  ½ in het schema.
- **Een backtest die er mooi uitziet.** Als de cijfers slecht zijn, krijg je de
  slechte cijfers. Dat is het hele punt van Fase 5.
