# Hypothese: marktsoorten verdienen een andere voorkeursvolgorde

## Vooraf geregistreerde verwachting

Dezelfde chartvorm heeft niet in iedere markt dezelfde betekenis. Een gap in
een aandeel, 24/7-liquiditeit in crypto en sessie-overlap in FX zijn andere
microstructuren. Eén universele signaaldrempel kan daarom bruikbaar blijven,
maar de volgorde waarin geldige setups worden behandeld hoort rekening te
houden met marktsoort, regime en de theorie die het signaal produceerde.

## Economische rationale

- FX en metalen worden sterk door sessie-liquiditeit en relatieve richting
  gedreven; trend en sweep zijn beide plausibel.
- Aandelen hebben discrete beursuren en gaps; H1/M15-reversals zijn minder
  betrouwbaar zonder aanvullend bewijs dan voortzetting met sectorsteun.
- Indices weerspiegelen brede risicobereidheid; marktbreedte is relevante
  context.
- Crypto handelt continu, maar weekenddiepte en volatility expansion maken
  momentum en liquidity events belangrijker dan een beurs-openpatroon.
- Commodities hebben eigen sessies en discontinuiteiten; trendcontinuatie
  krijgt voorrang boven een universele korte fade.

## Implementatiegrens

De assetklasse-policy mag uitsluitend de rangorde van reeds geldige setups
begrensd wijzigen. Hij mag nooit:

- een afgewezen setup goedkeuren;
- een scorethreshold verlagen;
- richting omdraaien;
- SL, TP, risico of lotsize aanpassen;
- een trade afdwingen om activiteit te produceren.

## Slagingscriterium

Minimaal 100 afgesloten trades per marktsoort en een onaangeraakte laatste 20%
van de data. Vergelijk de gerouteerde volgorde met de oude volgorde op netto R,
drawdown, calibration en long/short-balans. Een scherpe parameterpiek is falen.
