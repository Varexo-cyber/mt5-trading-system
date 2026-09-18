# Sectie 20: meelopen met de trend, bijkopen op de terugval

> Vooraf vastgelegd, vóór de eerste meting. De uitleg ná de uitslag schrijven
> is geen bewijs maar de definitie van data mining.

## Wat de eigenaar vroeg

> "Eén sectie die mee moet en kan met de trend, wanneer hij een pullback ziet
> een logische instap. Buyen 0.01 bij 4350, M1 M2 M3 M5 M15 bullish. Stel die
> zakt naar 4349.5 nog een buy 0.01, naar 4349.0 nog een, en zovoort. Elke 5
> min scalpen tot het weer teruggaat naar 4350; het moment dat het in winst
> staat uitcashen. Bij nieuws mag die niet instappen. Alle sessies meten."

## Wat dit ís, met de juiste naam

Dit is een **grid met averaging down**. Elke volgende order ligt op een
slechtere prijs dan de vorige, er is geen stop, en de uitstap is "wachten tot
het terugkomt".

Dat is geen oordeel over de eigenaar en ook geen weigering — het is de naam van
het mechanisme, en die naam bepaalt wat er gemeten moet worden. `risk_manager.
assert_not_forbidden` gooit hier vandaag een `ForbiddenStrategyError` op:

    XAUUSD: adding to an existing BUY position (#…) is averaging down /
    gridding — forbidden

Die regel staat er op verzoek van de eigenaar zelf. Deze studie haalt hem niet
weg. Meten is niet handelen, en de meting is precies wat de vraag beantwoordt.

## De bewering

Bij XAUUSD levert het opbouwen van een ladder longs tijdens een terugval,
binnen een opwaartse stapel van M1/M2/M3/M5/M15, een positieve netto verwachting
per mand — en een verlieskans die op de echte rekeninggrootte te dragen is.

## Wie staat er aan de andere kant?

**Dit is de zwakste schakel van de hypothese en dat hoort hier te staan.**

Bij de gemeten regels van dit project is er steeds een partij aan te wijzen die
gedwongen handelt: een short-gamma hedger die vóór settlement moet bijkopen,
een stop-cluster dat wordt geraakt. Hier is die partij er niet.

Een grid verdient niet aan een mechanisme. Hij verdient aan het feit dat prijs
vaker terugkeert dan doorschiet, en betaalt dat terug op de dag dat hij
doorschiet. De verwachting is dus niet positief omdát iemand anders verliest —
ze is positief zolang de steekproef geen trend bevat die groot genoeg is.

Dat is een houdbaarheidsvraag, geen voorsprongvraag. Daarom staat de ruïnekans
hieronder boven de trefkans.

## Voorspellingen, vooraf uitgesproken

- Trefkans per mand: **90–99 %.** Een hoge trefkans is hier geen bewijs van
  iets; hij is een eigenschap van het mechanisme.
- Gemiddelde winst per mand: klein en positief.
- Aantal manden per jaar: 200–600.
- **Diepste onderwaterstand van één mand: dit is het getal dat telt.**
- **Waar dit hoort te falen:** in elke aanhoudende daling. Eén trend van
  voldoende omvang wist meer uit dan een jaar aan manden opbouwt.

## Wat er gemeten en gerapporteerd moet worden

Een grid meet je niet met trefkans en totaalwinst. Die twee zien er bij elk
grid goed uit tot het moment dat de rekening weg is — dat is niet cynisch maar
de vorm van de uitkomstverdeling. Daarom is de rapportage omgedraaid:

1. **Ruïnes.** Hoe vaak zou de rekening bij deze startbalans en dit lot zijn
   opgeblazen? Eén is te veel.
2. **Diepste onderwaterstand per mand**, in euro en in punten.
3. **Meeste benen in één mand.** Hoe diep werd de ladder ooit?
4. **Langste tijd onder water.** Kapitaal dat vastzit is kapitaal dat niets doet.
5. **De ergste losse mand**, voluit.
6. Pas dáárna: trefkans, netto resultaat, per kalenderjaar, per sessie, per uur.

## Twee varianten, want de ene is te handelen en de andere niet

| | benen | mandstop |
|---|---|---|
| `open` — wat er gevraagd is | onbegrensd | geen |
| `begrensd` — wat er te handelen valt | max N | bij −X euro dicht |

De tweede bestaat om te laten zien wat de begrenzing kost. Levert `begrensd`
bijna hetzelfde op zonder ruïnes, dan is de onbegrensde versie nergens voor
nodig. Levert `begrensd` niets op, dan zat de hele winst in de staart die de
rekening opblaast, en dan is dat het antwoord.

## Testopzet

- Instrument: XAUUSD (ontwikkelset). XAUJPY en US30 blijven achter voor de
  kruistest.
- Klok: M1 voor de laddervulling, M1/M2/M3/M5/M15 voor de stapelrichting.
- Vulling **pessimistisch**: binnen één bar worden eerst alle bereikte
  ladderprijzen gevuld en pas daarna wordt op winst gekeken. Raakt een bar
  zowel een nieuw been als het doel, dan wint het been.
- Kosten: spread twee keer per been (in en uit).
- Nieuwsvenster uitgesloten via de bestaande kalenderfilter.
- Verdeling per kalenderjaar, per sessie (Asia/Londen/New York/overlap) en per
  uur — alle drie apart, want dat is wat gevraagd is.

### Geveegde parameters

| parameter | waarden |
|---|---|
| stap tussen benen | 0,5 · 1,0 · 2,0 punten |
| max benen | ∞ · 5 · 10 · 20 |
| mandstop | geen · −2 % · −5 % van de balans |
| lot per been | 0,01 · 0,49 (wat hij werkelijk handelt) |

**Aantal configuraties: 3 × 4 × 3 × 2 = 72.** Dat getal hoort bij de uitslag,
want de beste van 72 is iets anders dan een ontdekking.

## Uitslag

In te vullen na afloop. Link naar `docs/modules/section_twenty.md`.

- [ ] ≥100 manden
- [ ] Nul ruïnes op de echte balans
- [ ] Breed parameterplateau, geen piek
- [ ] Elk kalenderjaar apart positief
- [ ] Houdt stand op een instrument waarop hij niet ontwikkeld is

## Oordeel

`weight 0, kept for research` tot alle vijf de vakjes aangevinkt zijn. De
verboden-assertie blijft tot dat moment staan zoals hij staat.
