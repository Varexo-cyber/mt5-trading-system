# Hypothese: langdurige uitkomsten kunnen selectie begrensd kalibreren

## Vooraf geregistreerde verwachting

Losse Claude-weigeringen en vijf trades zijn anekdotes. Herhaalde, gerealiseerde
uitkomsten binnen dezelfde assetklasse, setupfamilie, horizon, richting en
regime kunnen na een grote steekproef wel informatie geven over welke van twee
reeds geldige kandidaten eerst aandacht verdient.

## Guardrails

- Alleen werkelijk afgesloten trades beïnvloeden de live-rangorde.
- Counterfactuals beoordelen filters en Claude, maar krijgen geen live-gezag.
- Minder dan 40 vergelijkbare trades geeft exact nul aanpassing.
- De schatting wordt naar het accountgemiddelde teruggetrokken.
- De aanpassing is begrensd en wijzigt alleen rangorde.
- Risico, lotsize, SL, TP, thresholds en modulegewichten blijven onaantastbaar.
- Neon-uitval levert een neutrale nul op en blokkeert nooit de handelslus.

## Anti-recency

De schatting gebruikt de volledige accountgeschiedenis en wordt op
steekproefgrootte gewogen. Een recente winnaar of verliesreeks kan daardoor
niet zelfstandig de selectie omgooien.

## Slagingscriterium

Pas na minimaal 100 afgesloten trades per geëvalueerde segmentfamilie:

- calibration op een onaangeraakte tijdsplit;
- netto verbetering tegenover de ongekalibreerde rangorde;
- geen verslechtering van maximale drawdown;
- parameterplateau rondom minimumsteekproef en shrinkage;
- afzonderlijke rapportage van long en short.
