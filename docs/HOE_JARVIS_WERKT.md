# Hoe Jarvis werkt — heel eenvoudig uitgelegd

## Eerst het eerlijke korte antwoord

Jarvis kan automatisch echte orders naar jouw gekoppelde Eightcap-account
sturen, maar alleen in **EXPERIMENTAL LIVE** en alleen als alle beveiligingen
groen zijn. Het is geen menselijk bewustzijn en Codex zit niet permanent in je
computer. De vaste handelssoftware neemt de meeste beslissingen. Claude krijgt
pas aan het einde een klein, gecontroleerd voorstel en mag alleen goedkeuren of
afwijzen.

Er is geen beloofd winstpercentage. Op dit moment is er ook nog geen eerlijke
live steekproef waarmee een betrouwbaar succespercentage kan worden genoemd.
Een systeem met nul trades heeft geen bewezen winstkans.

## Wat gebeurt er wanneer je de pc aanzet?

1. Windows start MetaTrader 5 geminimaliseerd.
2. MT5 logt in op het opgeslagen Eightcap-account.
3. Na ongeveer 60 seconden start Windows Jarvis in de ingestelde modus.
4. Na ongeveer 90 seconden opent Windows het lokale dashboard.
5. Jarvis controleert of MT5 verbonden is en of het rekeningnummer, de server en
   de valuta precies overeenkomen met het experimentele contract.
6. Als het bestand `STOP` bestaat, plaatst Jarvis niets. Hij sluit alleen zijn
   eigen nog openstaande posities en stopt daarna.
7. Als STOP vrij is, begint de scanronde.

MT5 is dus de kabel naar Eightcap. Jarvis is de bestuurder. Het dashboard is
het scherm en de noodrem.

## Wat scant hij precies?

Jarvis vraagt de volledige symboolcatalogus op die Eightcap via jouw MT5 laat
zien. Dat betekent niet dat elk symbool op dat moment open, bruikbaar of
betaalbaar is.

Hij screent in iedere ronde alle ondersteunde markten uit de Eightcap-catalogus.
Het systeem probeert iedere 30 seconden een nieuwe ronde te beginnen. Heeft MT5
langer nodig om alle symbolen te behandelen, dan wacht Jarvis totdat die ronde
klaar is en begint hij daarna meteen opnieuw. Rondes overlappen nooit.

Alle markten worden dus meegenomen, maar niet alle 847 markten worden iedere
seconde op alle timeframes doorgerekend. Dat zou MT5 onnodig belasten en veel
slechte of halve data opleveren.

Bij iedere markt controleert hij eerst goedkoop:

1. Is dit contract bij de broker verhandelbaar?
2. Herkent Jarvis de assetklasse, zoals forex, crypto, aandeel of index?
3. Is er een geldige en verse prijs, of is de markt gesloten?
4. Is de spread niet te duur?
5. Zijn er genoeg H1-candles?
6. Kan volatiliteit (ATR) worden berekend?
7. Hoe sterk en actief lijkt deze markt vergeleken met de rest van deze groep?

Iedere goedkope-scan-overlever mag door naar de dure analyse, tot aan de ruime
technische bovengrens van 2.000. Bij de huidige Eightcap-catalogus betekent dit
dat niet alleen een vooraf gekozen top-5 wordt gelezen. In het tabblad **Live
scanner** zie je voor iedere inspectie het antwoord en de afwijsreden.

## Wat gebeurt er in de diepe analyse?

Voor iedere overlevende kandidaat haalt Jarvis gesloten candles op van W1, D1,
H4, H1, M15, M5 en M1. Een candle die nog bezig is telt niet mee. Daarna kijkt de vaste
analyse-engine naar de kleine, vooraf bepaalde kern van signalen:

- marktstructuur;
- trend en momentum;
- liquidity sweep;
- reactie op een prijsniveau;
- volatiliteitsregime als veiligheidscontrole.

Hij probeert dus niet letterlijk iedere tradingtheorie van internet. Dat zou
onmeetbaar en extreem gevoelig voor overfitting zijn. Alleen expliciet gebouwde
en testbare regels tellen mee.

De grafieken hebben niet allemaal hetzelfde stemrecht. Een H1-structuur- of
momentumsignaal is een swingplan: D1/W1 mogen daar zwaar tegenin gaan. Een losse
M15-liquidity-sweep is een intradayplan: stop en target komen van M15 en een
tegengestelde D1-trend is alleen achtergrond. Pas als de nabije hogere
timeframes samen duidelijk tegenwerken, wordt die korte trade geblokkeerd. Dit
werkt exact hetzelfde voor BUY en SELL.

Daarna vergelijkt Jarvis geldige kandidaten per marktsoort. Forex krijgt
relatieve valutasterkte, aandelen en indices marktbreedte, crypto zijn eigen
24/7-groep en metalen/commodities hun eigen groep en modulevoorkeur. Dit bepaalt
alleen welke geldige kans eerst bekeken wordt; het kan geen afgewezen trade
toveren.

## Wanneer mag een echte trade worden geplaatst?

Een kandidaat moet daarna door alle poorten:

1. Het contract is op dit account verhandelbaar en de echte lotsize past binnen
   het actieve experimentele contract; een handgeschreven vier-symbolenlijst is
   niet langer de catalogusgrens.
2. De actuele positie- en blootstellingsgrenzen geven ruimte.
3. Dagverlies, weekverlies en totale drawdown zitten onder hun grens.
4. Het nieuwsfilter, sessiefilter, spreadfilter en correlatiefilter keuren goed.
5. De berekende stoploss komt uit de marktstructuur en volatiliteit.
6. De take-profit voldoet minimaal aan de ingestelde reward/risk-verhouding.
7. De brokerlotgrootte kan echt binnen het actieve rekeningrisico vallen. Kan
   0,01 lot dat niet, dan wordt de trade overgeslagen; Jarvis rondt nooit omhoog.
8. MT5 bevestigt dat er genoeg marge is.
9. Claude ontvangt het begrensde tradevoorstel en mag alleen `approve` of `veto`
   antwoorden. Geen antwoord, fout antwoord of API-storing betekent geen trade.
10. Vlak voor verzenden controleert Jarvis STOP en de accountgrenzen nogmaals.

Pas daarna stuurt Jarvis de order met stoploss en take-profit naar MT5. MT5
stuurt hem naar Eightcap. Een hoge leverage verandert het risicocontract niet;
leverage verlaagt alleen de vereiste marge.

## Wat gebeurt er na een order?

Jarvis controleert zijn eigen positie opnieuw bij iedere cyclus. Hij kan de
stoploss beheren, nieuws-de-risking toepassen en sluiten wanneer een
veiligheidsregel dat verlangt. Hij vergelijkt bovendien MT5 met zijn journaal,
zodat een onverwacht gesloten order niet stil verdwijnt. Handmatige posities
zonder Jarvis' magic number worden door de noodstop niet aangeraakt.

Tussen volledige scans controleert de snelle bewaker ongeveer iedere seconde.
De gezondheidslezing gebruikt uitsluitend **afgesloten** candles: de lopende
candle mag via de verse tick waarschuwen, maar kan niet zelfstandig doorgaan
voor een bevestigde structuurbreuk. Forex gebruikt M1/M5; crypto, aandelen en
langzamere CFD-producten krijgen hun eigen, meestal tragere profiel. De vaste
avondsluiting geldt alleen voor forex en wordt niet meer blind op aandelen,
metalen of commodities toegepast.

Claude mag een open positie alleen minder riskant maken. Een niet-HOLD-advies
onder de ingestelde confidence-drempel wordt hard geweigerd. Ook wordt een door
Eightcap geweigerde sluitorder nooit meer als een gesloten positie geboekt:
Jarvis vermeldt expliciet dat hij nog openstaat en blokkeert bij een onopgeloste
reconciliatiefout nieuwe entries.

Claude krijgt bij zo'n beheerbeoordeling niet alleen de actuele candles. Hij
krijgt ook het oorspronkelijke instapplan, de module-redenen en Claude-review
van de entry, de oorspronkelijke SL/TP, de hoogste bereikte winst in R en alle
eerdere beheeracties. Daardoor kan hij expliciet bepalen of de **oorspronkelijke
trade-these** nog intact, verzwakt of ongeldig is. De normale beoordeling is om
de 15 minuten; een verslechterende gezondheidslezing, een nieuwe winstzone of
het teruggeven van winst kan hem vanaf twee minuten na de vorige beoordeling
eerder wakker maken. De goedkope lokale bewaker blijft intussen ongeveer iedere
seconde werken. Er wordt dus niet iedere seconde een dure, inconsistente
Claude-call gedaan.

Iedere analysecyclus, afwijzing, orderpoging, fill, fout en sluiting wordt
opgeslagen. Rapporten beschrijven dus ook waarom er géén trade kwam. SQLite is
het lokale uitvoeringsdagboek. Als `NEON_DATABASE_URL` op de VPS is ingesteld,
worden beslissingen, trades, beheeracties, lessen, nieuws en Claude-weigeringen
ook duurzaam in Neon opgeslagen.

Een geweigerd maar uitvoerbaar plan wordt later passief gevolgd: zou de oude
SL of TP eerst geraakt zijn? Zo leert het systeem of Claude en de filters vaak
terecht weigerden. Die denkbeeldige trades worden nooit vermengd met echt geld.
Pas na minimaal 40 werkelijk afgesloten, vergelijkbare trades mag de historie
de volgorde van nieuwe geldige kandidaten een klein beetje wijzigen. Bij minder
data is de invloed exact nul; risico, lotsize, SL, TP en toelatingspoorten
blijven ongewijzigd.

## Zo zet je hem nu weer aan

1. Laat MT5 open en ingelogd op het juiste Eightcap-account.
2. Open `http://127.0.0.1:8501` en ga naar **Control**.
3. Typ exact `clear stop` in het bevestigingsveld.
4. Klik **CLEAR STOP + START REAL TRADING**.
5. Wacht enkele seconden. Bij **Jarvis service** moet `RUNNING` verschijnen.
6. Open **Live scanner**. Na de eerste voltooide cyclus verschijnen de markten
   met hun redenen. Op een groot brokeraccount kan die eerste cyclus even duren.

Wil je alleen de noodstop wissen en nog niet handelen, kies dan **Clear STOP
only**. Daarna kun je MONITOR, PAPER of EXPERIMENTAL LIVE apart starten.

## Zo zet je hem uit

Klik in **Control** op **STOP AND FLATTEN JARVIS**. Dan gebeurt dit:

1. Er komt direct een duurzaam STOP-bestand.
2. Nieuwe orders worden onmiddellijk geblokkeerd.
3. Jarvis probeert alleen zijn eigen posities te sluiten.
4. Na uiterlijk ongeveer één scaninterval controleert hij dit opnieuw en stopt
   het proces zodra zijn posities dicht zijn.
5. De STOP blijft ook na opnieuw opstarten bestaan totdat jij hem bewust wist.

Alleen het browservenster sluiten zet Jarvis niet uit. Alleen MT5 sluiten is ook
geen nette stopmethode; gebruik de rode noodstop.

## Wat zie je in het dashboard?

- **Overview**: balans, equity, verbinding en snelle marktinformatie.
- **Live scanner**: wat zojuist is bekeken en waarom iets door mocht of afviel.
- **AI exchange**: het exacte veilige voorstel aan Claude, `PENDING` tijdens de
  aanvraag, Claude's antwoord en kandidaten die vóór Claude zijn afgewezen.
- **Charts**: door jou gekozen markt en maximaal vier timeframes tegelijk.
- **Positions**: open broker- en paperposities, iedere beheeractie, de volledige
  tijdlijn per trade en de passieve vergelijking met de oorspronkelijke SL/TP.
- **PDF report**: een momentopname die je kunt downloaden.
- **Control**: starten, harde STOP, Claude-status en de laatste heartbeat.

Het dashboard ververst de scannerweergave iedere vijf seconden. De scanner zelf
werkt ongeveer iedere 30 seconden; vijf seconden verversen maakt de data dus
niet vijf seconden nieuw.

Claude wordt niet betaald aangeroepen voor iedere catalogusinspectie. Eerst
moeten analyse, whitelist, rekeningrisico, nieuws, sessie, spread, correlatie,
lotgrootte en marge allemaal groen zijn. Dit beschermt het kleine API-tegoed.
Wanneer dat gebeurt, verschijnt het verzoek direct in **AI exchange** en het
antwoord normaal enkele seconden later. Geen antwoord of een API-fout is altijd
een veto.
