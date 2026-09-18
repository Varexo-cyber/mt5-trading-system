# Sectie 21: twee benen openen, de verliezer eruit, de winnaar laten lopen

> Vooraf vastgelegd, vóór de eerste meting.

## Wat de eigenaar vroeg

> "Hedging op mijn balans, bijvoorbeeld 0.01 buy en 0.01 sell. Waar de trend
> mee gaat holden, en waar het in verlies staat close."

Met als voorbeeld het myfxbook-account `Aurix/aurix-pulse`.

## Wat dit ís, met de juiste naam

Dit is **geen hedge maar een straddle**. Een hedge dekt bestaand risico af; hier
wordt risico gecreëerd in twee richtingen tegelijk en daarna één kant
weggegooid. Je dekt niets af — je betáált om te ontdekken welke kant het op
gaat.

Netto-blootstelling is nul tot het moment dat één been sluit. Vanaf dat moment
sta je gewoon directioneel, met de kosten al gemaakt. De breakeven-identiteit
ligt daarmee vast en is niet te ontlopen:

    netto = winst winnaar − verlies verliezer − 2 × spread × 2 benen

`risk_manager.assert_not_forbidden` weigert dit vandaag:

    opening BUY while #… is SELL would hedge the position — forbidden;
    close it instead

Die regel blijft staan. Meten is niet handelen.

## Wat het voorbeeldaccount er wél over zegt

Nagerekend uit zijn eigen statistieken, niet overgenomen:

| | |
|---|---|
| Trades | 3.497 |
| Trefkans | 56,5 % (1976/1521) |
| Gemiddelde winst | $31,47 |
| Gemiddeld verlies | −$24,71 |
| Profit factor | 1,65 (nagerekend: 1,65) |
| Verwachting | +$7,03 (opgegeven +$7,04) |
| t-waarde | **6,76** — de voorsprong is echt |
| Absolute gain | **+49,4 %**, niet de +495,7 % in de kop |

**Dit is geen grid- of martingaleprofiel.** Die hebben een trefkans boven de 90 %
met winsten die een fractie van de verliezen zijn. Hier is de gemiddelde winst
grόter dan het gemiddelde verlies bij 56 % raak: een gewone scalper met een
bescheiden, statistisch echte voorsprong.

**Maar er wordt wél gestraddled.** Op 28 augustus staat de beste trade op
+6.677 pips en de slechtste op −6.598 pips. Zelfde dag, bijna spiegelbeeldig:
dat is een paar dat wordt afgewikkeld.

**En het rekensommetje dat de omvang begrenst.** Zou élke cyclus precies één
winnaar en één verliezer opleveren, dan was de trefkans mechanisch exact 50 %.
Hij staat op 56,5 %. Het mechanisme kan dus hooguit een deel van zijn boek
verklaren, en zijn voorsprong zit voor een deel ergens anders.

## De bewering

Twee benen openen en de verliezer bij −X punten afkappen levert op XAUUSD een
hogere netto verwachting per cyclus dan hetzelfde in- en uitstapmoment met één
vooraf gekozen richting.

## Wie staat er aan de andere kant?

**Niemand, en dat is het probleem met deze hypothese.**

Een straddle verdient niet aan een tegenpartij die gedwongen verliest. Hij
verdient — als hij verdient — aan de vórm van de bewegingsverdeling: dikke
staarten. Je betaalt een vast bedrag (X + de spreads) voor de kans op een
beweging die groter is dan dat.

Dat is een verdelingsvraag, geen voorsprongvraag. En verdelingsvragen zijn
zonder tegenpartij zelden houdbaar, want er is niemand die er structureel voor
betaalt.

## Voorspellingen, vooraf uitgesproken

- Trefkans per cyclus: 35–50 %. Lager dan bij één richting, want de kosten
  liggen hoger.
- Netto per cyclus: **rond nul of licht negatief.** Dat is de voorspelling.
- Cycli per dag: 4–8, in lijn met de ~6 per dag van het voorbeeldaccount.
- **Waar dit hoort te falen:** in een rustige markt. Elke wiebel kapt het
  verkeerde been af en je betaalt twee spreads voor niets.

## De controle die het beslist

Dit is het hele punt van deze sectie, en zonder deze vergelijking zegt de
uitslag niets:

| | |
|---|---|
| `straddle` | beide benen, verliezer afkappen |
| `één richting` | **zelfde moment, zelfde X en Y, één kant** |
| `altijd long` | de domme nulhypothese, want goud steeg |

Verslaat `straddle` die tweede niet, dan is het extra been **zuivere
kostenpost** en is de vraag beantwoord zonder dat er iets live hoeft. Zijn
voorsprong zit dan in de instap, niet in de hedge.

## De val bij "welk uur werkt het best"

De eigenaar vraagt hier bij elke sectie om, en het is een redelijke vraag met
een gevaarlijk antwoord: **het beste van 24 uren is het maximum van 24 ruizige
getallen.** Dat ziet er altijd goed uit, ook als er niets is.

Daarom hoort er bij elke uursuitsplitsing een permutatietoets: gooi dezelfde
cycli willekeurig over de uren heen, duizend keer, en kijk hoe goed het beste
uur er dán uitziet. Is het echte beste uur niet beter dan die verdeling, dan is
het geen vondst maar een selectie.

Hetzelfde geldt voor sessies, met 4 hokjes in plaats van 24.

## Testopzet

- Instrument: XAUUSD (ontwikkelset), M1-bars.
- Instap: elke N minuten, alle uren, met en zonder nieuwsfilter.
- **Beide benen afgekapt in één bar** wanneer het bereik van die bar allebei de
  drempels omvat. Binnen een bar is de volgorde onbekend, en de pessimistische
  lezing is dat je ze allebei kwijt bent.
- Kosten: spread twee keer per been, dus vier keer per cyclus.

### Geveegde parameters

| parameter | waarden |
|---|---|
| X — verliezer afkappen | 2 · 5 · 10 · 20 punten |
| Y — doel winnaar | 5 · 10 · 20 · 40 punten |
| stop winnaar | geen · 10 · 20 punten |

**Aantal configuraties: 4 × 4 × 3 = 48**, en dat getal hoort bij de uitslag.

## Uitslag

In te vullen na afloop.

- [ ] ≥100 cycli
- [ ] Verslaat `één richting` op dezelfde momenten
- [ ] Verslaat `altijd long`
- [ ] Het beste uur overleeft de permutatietoets
- [ ] Elk kalenderjaar apart positief

## Oordeel

`weight 0, kept for research` tot alle vijf de vakjes aangevinkt zijn.
