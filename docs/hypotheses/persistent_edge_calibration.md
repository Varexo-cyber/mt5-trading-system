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

## Uitbreiding vooraf geregistreerd: setup-lifecycle en thesis-invalidation

Waargenomen fout op 20 augustus 2026: richting en instapmoment waren nog te
veel één beslissing. ETHUSD kon terecht bullish zijn en toch bovenaan een live
spike worden gekocht. Een `WAIT_RETEST` werd later als een nieuwe anonieme
setup bekeken in plaats van als de volgende toestand van dezelfde thesis. Ook
domineerde de ruwe confluencescore de begrensde leerlaag, terwijl de scorecard
geen monotone relatie tussen ruwe score en gerealiseerde R liet zien.

Voor implementatie wordt daarom vastgelegd:

1. Bewaar een doorgeschoten of actief terugtrekkende setup over scancycli.
2. Eis na doorschieten eerst een meetbare pullback en daarna een nieuw gesloten
   directioneel instapframe voordat een order mag worden gepromoveerd.
3. Laat iedere setup verlopen op zijn eigen horizon; een oude thesis wordt
   nooit permanente entrytoestemming.
4. Bewaar iedere detectie en overgang in de decision-context. De lifecycle
   vertraagt een entry en wist geen setup uit de telling.
5. Beperk de dominantie van de ruwe score in de rangorde en geef alleen
   begrensde extra prioriteit aan onafhankelijke bewijsfamilies. Geen van beide
   mag een broker-, risk-, cost- of executiongate omzeilen.
6. Een materieel verliezende positie is inhoudelijk ongeldig wanneer minimaal
   twee onafhankelijke healthfamilies overeenkomen, minstens één daarvan
   chartbewijs is, en de gezamenlijke read al `deteriorating` is. Dit mag alleen
   risico verminderen: sluiten, nooit bijkopen of de stop verbreden.

Succes is netto R en entry-MAE op onaangeraakte forward trades, niet alleen
trade count of winrate. Setupdetecties vóór timing blijven zichtbaar, zodat
minder orders nooit als betere setupdetectie kunnen worden verkocht. De
managementbaseline tegen het oorspronkelijke SL/TP-plan mag niet verslechteren.

### Forward-amendement 21 augustus 2026: geen dubbele recovery-poort

De eerste lifecycleversie kon een setup in `WAIT_PULLBACK` of
`WAIT_RESUMPTION` houden nadat de gewone entry-qualitymeting alweer
`ENTER_NOW` gaf. Daarmee moest dezelfde herstelde entry tweemaal bewijzen dat
zij tijdig was: eerst via range, EMA-afstand, candlebody, live gap en adverse
bar, daarna opnieuw via vaste lifecycle-ATR-drempels. De lifecycle blijft het
geheugen van een te late thesis en iedere wait blijft met zijn onaangeraakte
counterfactual in het journal staan. Zodra de oorspronkelijke entry-quality
op een latere cyclus opnieuw `ENTER_NOW` geeft, mag de lifecycle die entry niet
meer blokkeren. Deze wijziging wordt uitsluitend op nieuwe forward trades
beoordeeld.
