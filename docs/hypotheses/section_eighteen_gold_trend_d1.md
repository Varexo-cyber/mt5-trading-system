# Hypothesis: section_eighteen_gold_trend_d1

> Ingevuld **voordat** er iets gemeten is. Dit is het eerste document in dit
> project dat in die volgorde is geschreven, en dat is precies waarom alle
> voorgaande cijfers omvielen.

## Waarom deze sectie bestaat, en waarom nu

Op 12 september 2026 gaf `domtest.cmd holdout` dit, over 1 sep 2024 - 31 aug
2025, het jaar dat het systeem nooit gezien had:

    de hele dag long goud, 256 dagen        +1250,93 R
    sectie zes over dezelfde periode           +4,21 R

Goud maakte een van de grootste bewegingen van zijn geschiedenis. Sectie zes is
een LONG-ONLY goudstrategie. Hij stond de goede kant op en ving er drie tiende
van een procent van.

Dat is geen kostenprobleem en geen modelprobleem. **Het is een
deelnameprobleem.** Alle 44 modules zijn intraday: ze zijn per definitie
afwezig op het moment dat een beweging van maanden zich voltrekt. Ze zoeken
richting die in dit geval gratis was.

Deze sectie is de eerste in dit project die op de dagklok handelt.

## Claim

XAUUSD kent aanhoudende opwaartse trends over weken tot maanden. Een positie
die long gaat zodra de slotkoers boven een traag gemiddelde staat en blijft
zitten tot dat gemiddelde breekt of een brede stop raakt, vangt een meetbaar
deel van die beweging, tegen een fractie van de kosten van intraday handelen.

**LONG ONLY.** Niet uit voorkeur maar uit ontwerp: dit is een deelnamevoertuig,
geen richtingvoorspeller. Short gaan zou een tweede hypothese zijn die apart
bewijs verdient.

## Economische rationale — wie verliest hier geld, en waarom blijft hij dat doen

Twee partijen, en allebei zijn ze structureel en niet dom:

1. **Wie moet verkopen in een stijgende markt.** Centrale banken die hun
   reserves herwegen, mijnbouwers die productie afdekken, fondsen die op
   gewicht herbalanceren. Die verkopen omdat hun mandaat dat zegt, niet omdat
   ze denken dat goud gaat dalen. Dat is prijsongevoelige aanbod tegen een
   koper die kan wachten.
2. **Wie te vroeg winst neemt.** De particulier die na +5% verkoopt "omdat het
   al zo hard is gegaan". Dat is dezelfde onderreactie die de literatuur voor
   time-series momentum als mechanisme aanwijst.

Waarom blijft dat bestaan: het is oncomfortabel. Trendvolgen betekent lange
periodes van niets doen, een lage trefkans, en meerdere valse starts per echte
beweging. Dat is een gedragsprijs, geen inefficiëntie die weggearbitreerd wordt.

**Waar dit mechanisme zwak is, en dat hoort hier ook:** ditzelfde verhaal is te
vertellen over elke markt die ooit gestegen is. Het is achteraf niet te
onderscheiden van "goud ging omhoog". Daarom staat er hieronder een expliciete
faalvoorwaarde die dat afvangt.

## Voorspellingen, vooraf vastgelegd

- Verwachte trefkans: **30-45%**. Laag, met opzet. Trendvolgen verdient aan
  weinig grote winnaars en betaalt aan veel kleine verliezers. Een trefkans
  boven de 55% zou betekenen dat dit iets anders doet dan wat het beweert.
- Verwachte gemiddelde R: **+0,15 tot +0,50 per trade.**
- Verwacht aantal signalen: **8 tot 25 per jaar** op XAUUSD.
- Verwachte gemiddelde houdduur: **5 tot 30 kalenderdagen.**
- Verwachte terugval: **20 tot 60 R** over een jaar. Dit is de belangrijkste
  voorspelling, want de reden dat kopen-en-vasthouden onbruikbaar is, is niet
  het rendement (+1250 R) maar de terugval (295 R).

### Waar dit MOET falen

Deze regels zijn vooraf vastgelegd en er wordt achteraf niet aan gedraaid:

1. **In een zijwaartse markt.** Een reeks valse starts is de normale prijs. Als
   dat NIET gebeurt is er iets mis met de meting.
2. **Als het minder dan de helft haalt van "de hele dag long" op
   rendement-per-terugval.** Dat is de eigenlijke lat: buy-and-hold gaf in het
   holdoutjaar 1250,93 / 295,08 = **4,24**. Deze sectie moet boven **2,12**
   uitkomen of hij voegt niets toe aan gewoon goud kopen. Dit is de
   faalvoorwaarde die "het is gewoon de goudtrend" afvangt.
3. **Als de trefkans boven de 55% ligt.** Dan volgt hij geen trend maar iets
   anders, en is de meting niet wat hij beweert.
4. **Als hij op de holdoutperiode negatief is.** Geen uitzonderingen, geen
   "maar het regime was".

## Testopzet, vooraf vastgelegd

- **Instrument (ontwikkeling):** XAUUSD, en verder niets.
- **Achtergehouden voor de kruistest:** NDX100, SPX500, BTCUSD. Pas aanraken
  nadat XAUUSD een uitspraak heeft opgeleverd.
- **Klok:** D1 voor het signaal en de stop.
- **Ontwikkelperiode:** 1 sep 2025 - 11 sep 2026 (het venster waarop al het
  andere in dit project ook gekozen is; deze sectie mag daar dus niets aan
  ontlenen wat als bewijs telt).
- **Onaangeroerde holdout:** 1 sep 2024 - 31 aug 2025. **Exact één keer
  bekeken.** Datzelfde jaar draagt de +1250,93 R en de +4,21 R hierboven.
- **Parameters:** twee. `trend_bars` (lengte van het gemiddelde) en `stop_atr`
  (breedte van de stop in dagelijkse ATR).
- **Aantal geteste combinaties:** **maximaal 6.** Drie waarden voor
  `trend_bars` (20, 50, 100) maal twee voor `stop_atr` (2,0 en 3,0). Meer is
  het niet, en de Bonferroni-lat voor zes kandidaten is **t > 2,64**.

## Waarom twee parameters en niet acht

`_bonferroni_t` uit dit project zegt wat elke extra knop kost. Bij zes
kandidaten is de lat t>2,64; bij honderd is hij t>3,48. Sectie zes had
veertig parameters en heeft nooit een eerlijke lat gehaald omdat niemand ooit
geteld heeft hoeveel er geprobeerd was.

Twee knoppen, zes combinaties, van tevoren opgeschreven. Dat is de hele reden
dat deze sectie iets kan aantonen en de vorige zeventien niet.

## Wat deze sectie NIET is

- Geen vervanging van sectie zes. Die staat er los van en dit oordeelt niet
  over hem.
- Geen kopen-en-vasthouden met extra stappen. Faalvoorwaarde 2 bestaat precies
  om dat te kunnen uitsluiten.
- Geen kandidaat voor live geld. Niet na de ontwikkelperiode, niet na de
  holdout. Pas na een VOORUIT gemeten periode die op dit moment nog niet
  bestaat.

## Resultaat

Nog niet gemeten. In te vullen na `sectie18.cmd`.

- [ ] ≥30 trades over beide periodes samen (lager dan de gebruikelijke 100,
      omdat dit een lage-frequentiestrategie is; daarom weegt de terugval en de
      verhouding tot buy-and-hold hier zwaarder dan de t)
- [ ] Breed parameterplateau, geen piek
- [ ] Rendement per terugval boven 2,12 (de helft van buy-and-hold)
- [ ] Trefkans tussen 30% en 55%
- [ ] Positief op de holdout, die precies één keer bekeken is
- [ ] Paired t boven 2,64 na correctie voor zes kandidaten

## Oordeel

Nog geen.
