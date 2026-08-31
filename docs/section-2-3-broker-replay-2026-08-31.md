# Secties 2 en 3: gecorrigeerde brokerreplay

Datum: 31 augustus 2026. Kapitaal: EUR 203. Risico: maximaal 2% per trade.
Bron: `market-history.sqlite3`, 180 dagen brokerbars.

## Correcties vóór de meting

- Een M1-beslissing gebruikte eerder de laatste M5-close als gesimuleerde
  tick. De replay gebruikt nu de fijnste beschikbare serie.
- De rapportage beweerde één positie per symbool, maar handhaafde alleen het
  totale slotplafond. De replay handhaaft nu beide regels.
- Spread, minimale lotgrootte, sizing, break-evenbeheer en maximaal vier
  posities zijn meegenomen. De acht extra livepoorten zijn niet nagebootst;
  die kunnen setups weigeren, maar een negatieve strategie niet positief
  maken.

## Resultaten

| Universum | Modules | Portefeuilletrades | Winrate | Resultaat | Geld |
|---|---|---:|---:|---:|---:|
| US30, NDX100, SPX500, GER40 | S2 + S3, alle gevraagde klokken | 128 | 72,7% | -14,74R | -EUR 54,36 |
| 11 FX-paren | S2 M15 + M30 | 15 | 26,7% | -3,62R | -EUR 12,50 |

Op indices verloor Order Block M1 afzonderlijk -8,82R over 107
portefeuilletrades. Alle zes sectie/klokcombinaties waren negatief wanneer ze
los over alle indexsignalen werden gemeten. Negen van elf FX-markten waren bij
EUR 203 niet betaalbaar binnen een redelijke kostenlimiet; de twee resterende
markten waren samen eveneens negatief.

## Besluit

Secties 2 en 3 blijven als shadowmodules actief om nieuwe gegevens te meten,
maar zijn verwijderd uit `live_enabled_modules`. Kosten-, nieuws- of
confirmationpoorten losser zetten is niet verantwoord: de replay zonder die
extra weigeringen is al negatief. Er is ook geen nieuwe sectie live gezet;
de brede zoektocht leverde geen kandidaat op die holdout, multiple-testing en
kapitaal/kostenaudit overleefde.
