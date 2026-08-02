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

De beste vijf uit de volledige catalogus mogen door naar de dure analyse. In het tabblad
**Live scanner** zie je voor iedere inspectie het antwoord en de afwijsreden.

## Wat gebeurt er in de diepe analyse?

Voor maximaal vijf kandidaten haalt Jarvis gesloten candles op van D1, H4, H1,
M15 en M5. Een candle die nog bezig is telt niet mee. Daarna kijkt de vaste
analyse-engine naar de kleine, vooraf bepaalde kern van signalen:

- marktstructuur;
- trend en momentum;
- liquidity sweep;
- reactie op een prijsniveau;
- volatiliteitsregime als veiligheidscontrole.

Hij probeert dus niet letterlijk iedere tradingtheorie van internet. Dat zou
onmeetbaar en extreem gevoelig voor overfitting zijn. Alleen expliciet gebouwde
en testbare regels tellen mee.

## Wanneer mag een echte trade worden geplaatst?

Een kandidaat moet daarna door alle poorten:

1. Het symbool staat op de micro-live whitelist. Met deze rekening zijn dat nu
   `EURUSD.i`, `GBPUSD.i`, `USDJPY.i` en `AUDUSD.i`.
2. Er zijn niet te veel open posities of trades vandaag/deze week.
3. Dagverlies, weekverlies en totale drawdown zitten onder hun grens.
4. Het nieuwsfilter, sessiefilter, spreadfilter en correlatiefilter keuren goed.
5. De berekende stoploss komt uit de marktstructuur en volatiliteit.
6. De take-profit voldoet minimaal aan de ingestelde reward/risk-verhouding.
7. De brokerlotgrootte kan echt binnen maximaal 1% rekeningrisico vallen. Kan
   0,01 lot dat niet, dan wordt de trade overgeslagen; Jarvis rondt nooit omhoog.
8. MT5 bevestigt dat er genoeg marge is.
9. Claude ontvangt het begrensde tradevoorstel en mag alleen `approve` of `veto`
   antwoorden. Geen antwoord, fout antwoord of API-storing betekent geen trade.
10. Vlak voor verzenden controleert Jarvis STOP en de accountgrenzen nogmaals.

Pas daarna stuurt Jarvis de order met stoploss en take-profit naar MT5. MT5
stuurt hem naar Eightcap. Een hoge leverage verandert de 1%-risicoregel niet;
leverage verlaagt alleen de vereiste marge.

## Wat gebeurt er na een order?

Jarvis controleert zijn eigen positie opnieuw bij iedere cyclus. Hij kan de
stoploss beheren, nieuws-de-risking toepassen en sluiten wanneer een
veiligheidsregel dat verlangt. Hij vergelijkt bovendien MT5 met zijn journaal,
zodat een onverwacht gesloten order niet stil verdwijnt. Handmatige posities
zonder Jarvis' magic number worden door de noodstop niet aangeraakt.

Iedere analysecyclus, afwijzing, orderpoging, fill, fout en sluiting wordt
opgeslagen. Rapporten beschrijven dus ook waarom er géén trade kwam.

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
- **Charts**: door jou gekozen markt en maximaal vier timeframes tegelijk.
- **Positions**: open broker- en paperposities.
- **PDF report**: een momentopname die je kunt downloaden.
- **Control**: starten, harde STOP, Claude-status en de laatste heartbeat.

Het dashboard ververst de scannerweergave iedere vijf seconden. De scanner zelf
werkt ongeveer iedere 30 seconden; vijf seconden verversen maakt de data dus
niet vijf seconden nieuw.
