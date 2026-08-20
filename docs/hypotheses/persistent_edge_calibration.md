# Hypothese: langdurige uitkomsten kunnen selectie begrensd kalibreren

## Vooraf geregistreerde verwachting

Losse Claude-weigeringen en vijf trades zijn anekdotes. Herhaalde, gerealiseerde
uitkomsten binnen dezelfde assetklasse, setupfamilie, horizon, richting en
regime kunnen na een grote steekproef wel informatie geven over welke van twee
reeds geldige kandidaten eerst aandacht verdient.

## Uitbreiding vooraf geregistreerd: ensemble-selectiebrein

Een enkel exact segment is op een kleine rekening bijna altijd leeg, terwijl
alleen `LONG` tegenover `SHORT` veel te grof is. Daarom wordt de rangorde ook
begrensd gekalibreerd op afzonderlijke, herbruikbare facetten: setupfamilie,
horizon, detector, assetklasse, regime, sessie en ruwe-scoreband. De facetten
worden met shrinkage samengevoegd; meerdere detectoren op dezelfde trade tellen
als één detector-facet en niet als meerdere onafhankelijke trades.

De verwachting is dat dit vooral de **volgorde** verbetert: een setup die past
bij meerdere historisch sterke omstandigheden bereikt eerder de schaarse
uitvoerings- of reviewcapaciteit. Het aantal gevonden setups hoort gelijk te
blijven. De laag mag een geldige setup niet ongeldig maken en een ongeldige
setup niet promoveren.

## Guardrails

- Alleen werkelijk afgesloten trades beïnvloeden de live-rangorde.
- Counterfactuals beoordelen filters en Claude, maar krijgen geen live-gezag.
- Minder dan de vooraf geconfigureerde steekproefvloer geeft exact nul
  aanpassing; de live-overlay gebruikt 15 met zware shrinkage, terwijl het
  slagingscriterium hieronder pas bij 100 vergelijkbare trades wordt beoordeeld.
- De schatting wordt naar het accountgemiddelde teruggetrokken.
- De aanpassing is begrensd en wijzigt alleen rangorde.
- Uitschieters worden voor deze selectieschatting in R begrensd; één foutieve
  fill of uitzonderlijke winnaar mag het geheugen niet domineren.
- Detectorfacetten worden gemiddeld voordat ze met andere facetten worden
  gecombineerd, zodat drie correlerende detectoren geen driedubbele stem zijn.
- De ruwe enginescore is een invoerfacette, geen kansschatting en geen autoriteit.
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
