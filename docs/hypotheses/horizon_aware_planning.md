# Hypothese: de handelsduur bepaalt welke grafiek gezag heeft

## Vooraf geregistreerde verwachting

Een signaal dat op M15 ontstaat, hoort zijn stop, haalbare target en hogere-
timeframecontrole niet automatisch van dezelfde H1/D1/W1-regels te krijgen als
een H1-swing. De economische reden is eenvoudig: een intraday terugkeer na een
liquidity sweep kan bestaan binnen een tegengestelde weektrend. De deelnemers
die kort na de sweep verkeerd staan, hoeven de weektrend niet te keren om hun
positie af te bouwen.

Daarom verwachten we dat horizonbewuste planning:

1. minder intraday-setups als onbetaalbare H1-swings vermomt;
2. niet meer stops binnen de normale ruis plaatst;
3. minder valide korte shorts blokkeert op alleen een D1- of W1-trend;
4. geen voordeel geeft aan long of short; de regel is volledig symmetrisch.

## Implementatiegrens

- H1-structuur en H4/H1-momentum blijven `swing`.
- Een standalone M15 liquidity sweep wordt `intraday`.
- Een swing houdt de bestaande D1/W1-veto.
- Een intraday-setup wordt pas hard geblokkeerd wanneer minimaal twee hogere
  timeframes sterk tegenwerken. Een enkele hogere trend blijft context.
- Alle berekeningen gebruiken uitsluitend gesloten candles.

## Wat deze hypothese niet beweert

Dit maakt een setup niet winstgevend. Het voorkomt alleen dat een korte trade
wordt beoordeeld alsof hij een meerdaagse trendomkeer moet volbrengen. De
uitkomst wordt afzonderlijk gelogd en moet daarna out-of-sample worden gemeten.

## Slagingscriterium

Minimaal 100 trades per horizon, met kosten, en vervolgens vergelijken:

- netto R per trade;
- stop-outs binnen 0,5 ATR;
- targetbereik binnen de geplande horizon;
- long/short-symmetrie;
- resultaat bij aangrenzende horizonparameters.
