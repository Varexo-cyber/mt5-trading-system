# Sectie 4 — Walk-forward index reversal

Bron: 180 dagen Eightcap-brokerhistorie. Kapitaal: EUR 203.

Het model leerde uitsluitend op de oudste 50%. Drempel, polariteit en target
werden gekozen op het volgende kwart. Het nieuwste kwart werd daarna eenmaal
geopend als holdout. De gekozen variant is SPX500/H1, contrair, stop 1 ATR en
target 1,5R.

- Validatie: 87 trades, +0,258R per trade, circa EUR +32,46.
- Onaangeraakte holdout: 92 trades, 44,6% winst, +0,074R per trade, EUR +12,02.
- Exacte Jarvis-replay met confluence, sizing en break-evenbeheer: 83 trades,
  80,7% winnende exits, +4,64R en EUR +15,20 in 45 dagen.
- Juli en augustus positief; maximale onderzoeksdrawdown -6,77R.

Alleen `SPX500` is toegestaan. Secties 2 en 3 blijven shadow omdat hun
gecorrigeerde brokerreplays negatief waren. De nieuwe sectie heeft een eigen
breaker: na minimaal twintig trades stopt hij bij meer dan 70% verliezen of
acht verliezen achter elkaar.

Dit is een positief experiment, geen winstgarantie. De live-vormsteekproef is
83 trades en +0,74 sigma; juli leverde 90% van het totaal. Nieuwe live-data
moet bevestigen of de edge standhoudt.
