# Wat er buiten dit project bestaat — 12 september 2026

Gevraagd: zoek alles wat er te vinden is aan manieren van handelen die door
anderen bewezen en door weer anderen nagerekend zijn, met hun mechanisme, en
kom met opties.

Wat hier staat is dus GEEN lijst met dingen die gaan werken. Het is een lijst
met dingen die het waard zijn om over te beslissen, elk met wat er echt over
bekend is, inclusief wat erop tegen is. Iets in deze lijst opnemen zegt niets
over of het geld verdient — dat weet niemand vooraf, en dat is het hele punt
van de rest van dit document.

## Het basispercentage, want dat hoort erbij

De vraag die hieraan voorafging was "hoe word ik rijk". Dan hoort dit erbij, en
niet als ontmoediging maar omdat je het nodig hebt om je eigen cijfers op waarde
te schatten.

**Taiwan, alle daghandelaren op de beurs, 1992-2006** (Barber, Lee, Liu, Odean).
Vijftien jaar, ongeveer 450.000 handelaren per jaar. Minder dan 1% haalde
betrouwbaar winst na kosten. Gemiddeld verloren ze 23,9 basispunten **per dag**,
na kosten. In veertien van de vijftien jaren was het totaal negatief.

**Brazilië, indexfutures, 2013-2015** (Chague, De-Losso, Giovannetti). 19.646
mensen begonnen. Van de circa 1.600 die het langer dan 300 sessies volhielden
verloor **97%** geld, en **1,1%** verdiende meer dan het minimumloon.

Dat zijn de eerlijke basiskansen, en ze zijn een van de best gedocumenteerde
dingen in dit hele document.

**Lees ze in twee richtingen.** De ene kant is duidelijk. De andere kant is dat
sectie zes in een jaar dat het systeem nooit gezien had een bruto edge van
+0,079 R per trade had, en in het jaar erna +0,152. Dat is een instap die
positief is op onaangeraakte data, twee keer. In de Braziliaanse steekproef zou
dat je in de bovenste procenten zetten — en het is nog steeds niet genoeg om er
op €450 geld mee te verdienen, omdat handelen 0,066 R kost.

Dat is niet dezelfde uitkomst als "het werkt niet". Het is: het werkt een beetje,
en een beetje is op deze rekeninggrootte niet genoeg. Dat is een ander probleem
en het heeft andere oplossingen dan nog een sectie.

## Eerst: de twee getallen waar alles hieronder tegenaan botst

### 1. Gepubliceerde effecten slijten. Gemeten, niet vermoed.

McLean en Pontiff (*Journal of Finance*, 2016) namen 97 voorspellers uit de
literatuur en keken wat ermee gebeurde nadat ze bekend werden:

    -26%  buiten de steekproef waarop ze gevonden waren
    -58%  na publicatie

Alles hieronder verdient dus meteen een korting. Een effect dat in een paper
+0,20 doet, is realistisch +0,08 tegen de tijd dat jij het handelt.

**En er staat een levend voorbeeld in deze lijst.** De *overnight drift* — grote
positieve rendementen in aandelenindices tijdens de Europese opening — is
gedocumenteerd door de Federal Reserve Bank of New York (Boyarchenko, Larsen,
Whelan, Staff Report 917) met een echt mechanisme: dealers die aan het eind van
de Amerikaanse dag voorraad overhouden, daar risico voor lopen en een korting
eisen die ze 's nachts terugverdienen. Sterk verhaal, sterk bewijs, top-instituut.

Diezelfde Fed publiceerde in juli 2026 *The Disappearing Overnight Drift*: over
januari 2021 tot december 2025 is het venster van 02:00-03:00 dat vroeger het
meest opleverde, gemiddeld ongeveer nul geworden.

Dat is niet een reden om de literatuur te wantrouwen. Dat is de literatuur die
zijn werk doet. Maar het is wel de reden dat "bewezen door anderen" nooit
hetzelfde is als "werkt nu, voor jou".

### 2. Het is niet je gebrek aan ideeën. Het is je omzet.

Novy-Marx en Velikov (*Review of Financial Studies*, 2016) rekenden voor een
groot aantal anomalieën door wat er na handelskosten van overblijft. Hun
conclusie in één zin:

> Anomalieën met minder dan **50% omzet per maand** houden na kosten een
> significant resultaat over; bij hogere omzet vrijwel geen.

Waar staat Jarvis?

    3.464 trades in 24 maanden  =  144 trades per maand
    dat is ongeveer 14.400% omzet per maand
    de grens waar het nog uitkomt is 50%

    Jarvis zit er een factor 289 boven.

Dit is exact hetzelfde antwoord dat je eigen `kosten.cmd` gisteravond gaf, maar
uit een compleet andere bron. Sectie zes verdient bruto +0,079 R per trade in
het holdoutjaar en betaalt 0,066 aan kosten. De literatuur zegt: bij deze
handelsfrequentie is dat de normale uitkomst, niet de pech.

**Daarom is de bruikbare vraag niet "welke tactiek nog meer" maar "welke tactiek
handelt minder".** Elke optie hieronder krijgt daarom een omzetregel.

### En hoe je 44 modules kunt hebben en toch één idee

| klok | modules |
|---|---|
| M5 | 14 |
| M15 | 10 |
| M1 | 7 |
| H1 | 4 |
| M30 | 3 |
| D1 | 1 |

34 van de 44 draaien op een half uur of sneller. Order blocks, impulsen,
retests, breakouts, VWAP, EMA-kruisingen, momentum — dat is allemaal
intraday-patroonherkenning op de prijs zelf. Er zit geen enkele module in die
langer dan een dag aanhoudt, geen enkele die twee markten tegen elkaar afzet, en
geen enkele die op een agenda handelt in plaats van op een grafiek.

Dat is niet 44 ideeën. Dat is één idee, 44 keer, in het duurste segment dat er
bestaat.

---

## De opties

Gerangschikt op wat ze kosten om te handelen, want dat is de bindende beperking.

### A. Intraday momentum — het eerste half uur voorspelt het laatste

**Claim.** Het rendement van het eerste half uur van de handelsdag (gemeten
vanaf de slotkoers van gisteren) voorspelt het rendement van het laatste half
uur.

**Bewijs.** Gao, Han, Li en Zhou, *Journal of Financial Economics* 2018 — een
van de drie belangrijkste finance-tijdschriften. S&P 500 ETF, 1993-2013.
Voorspellende R² van 1,6%, en 2,6% als je het twaalfde half uur meeneemt. Het
effect is sterker op volatiele dagen, op dagen met veel volume en op dagen met
grote macrocijfers. Het is teruggevonden in tien andere veelverhandelde ETF's.

**Mechanisme.** Late herbalancering: partijen die maar één keer per dag
handelen, en producten met hefboom die aan het eind van de dag hun blootstelling
moeten bijstellen, duwen in dezelfde richting als de opening.

**Voor jou.** NDX100 en SPX500. Eén trade per dag, dus 2.100% omzet per maand —
zeven keer minder dan Jarvis nu, maar nog steeds ruim boven de grens van
Novy-Marx.

**Wat erop tegen is.** R² van 1,6% is klein. Het paper eindigt in 2013; dat is
dertien jaar geleden en de McLean-Pontiff-korting is hier volledig van
toepassing. En jouw NDX-secties zijn al twee keer op deze markt omgevallen.

**Oordeel.** De sterkste papieren van de lijst, maar het handelt nog steeds te
vaak voor jouw kostenstructuur. De moeite waard om te meten, niet om op te
hopen.

### B. Pre-FOMC drift — de 24 uur voor een rentebesluit

**Claim.** Amerikaanse aandelen lopen op in de 24 uur vóór een geplande
FOMC-aankondiging.

**Bewijs.** Lucca en Moench, *Journal of Finance* 2015. Het rendement in die 24
uur verklaarde meer dan **80% van de volledige aandelenpremie** over zeventien
jaar. Ook zichtbaar in andere grote internationale indices. Niet in
staatsobligaties, en niet rond andere macrocijfers — dus het is specifiek, wat
een goed teken is.

**Mechanisme.** Onzekerheid die vlak voor de aankondiging wordt afgebouwd; wie
het risico in dat venster draagt, wordt daarvoor betaald.

**Voor jou.** SPX500 en NDX100. **Acht trades per jaar. 67% omzet per maand** —
dit is de enige optie in de hele lijst die onder of rond de grens van Novy-Marx
valt.

**Wat erop tegen is, en het is fataal voor bewijsvoering.** Acht trades per
jaar. Volgens je eigen `_bonferroni_t` heb je bij 300 trades al +0,136 R per
trade nodig om iets te kunnen aantonen. Bij acht per jaar heb je decennia nodig
voordat een resultaat iets betekent. Én recenter werk vindt dat het effect
kortstondig werd nadat de Fed persconferenties invoerde.

**Oordeel.** Structureel het beste passend bij je kosten, en tegelijk
onbewijsbaar op jouw tijdschaal. Dat is een eerlijke impasse, geen oplossing.

### C. Time series momentum — het meest nagerekende dat bestaat

**Claim.** Wat de afgelopen 1 tot 12 maanden steeg, stijgt de komende maand
door; wat daalde, daalt door. Over aandelenindices, valuta, grondstoffen en
obligaties.

**Bewijs.** Moskowitz, Ooi en Pedersen (2012), 58 futures, 1985-2009. Dit is de
meest nagerekende strategie in de hele literatuur.

**En precies daarom staat hier de eerlijke kanttekening.** Later werk vindt dat
het resultaat grotendeels wordt gedragen door het schalen op volatiliteit — de
positiegrootte — en niet door het momentumsignaal zelf. Zonder die schaling
levert het ongeveer hetzelfde op als gewoon kopen en vasthouden. Recentere
out-of-sample-evaluaties vinden Sharpes die in-sample rond 0,1-0,2 liggen en
buiten de steekproef negatief worden voor bijna elke parameterkeuze.

**Voor jou.** Maandelijks handelen = 100% omzet per maand. Dicht bij de grens.

**Oordeel.** Het bekendste idee op de lijst, en het bewijs is betwist op het
punt dat er precies toe doet. Maar het bevat wel het beste losse inzicht van
deze hele lijst, en dat staat hieronder als optie E.

### D. Goud in de Aziatische sessie

**Claim.** Goud loopt op tijdens de Aziatische uren en zakt in Londen en New
York. Eén bron noemt +0,029% voor 00:00-07:00 UTC met p<0,001 over 23 jaar.

**Bewijskwaliteit: zwak, en dat moet je weten.** De concrete getallen komen uit
handelsblogs, niet uit een tijdschrift. Wat er wél peer-reviewed is
(Iwatsubo, Watkins, Xu over intraday-seizoenspatronen in goud- en
platinafutures in Tokio en New York) gaat over liquiditeit, volatiliteit en
volume — niet over een verhandelbaar richtingseffect.

**Maar dit is wel de goedkoopste vraag die je kunt stellen, en hij raakt je enige
werkende sectie.** Sectie zes handelt goud tussen 20:00 en 02:00 UTC, **alleen
long**. Dat venster overlapt de Aziatische sessie.

Dus: is sectie zes een model, of heeft sectie zes een sessie-effect
heruitgevonden en er een model omheen gebouwd?

Dat is te beantwoorden op data die je al hebt, met de domst mogelijke
vergelijking: koop goud om 20:00 UTC, verkoop om 02:00, elke dag, geen model,
geen drempel, geen filter. Zet dat naast de 316 trades die sectie zes in het
holdoutjaar deed.

- Doet de domme versie het net zo goed → sectie zes is versiering en je hebt
  een veel simpeler ding met veel minder parameters om te overfitten.
- Doet de domme versie het slechter → het model verdient zijn plek, en dat is
  voor het eerst aangetoond.

Allebei de uitkomsten zijn winst. Dit is wat ik morgen als eerste zou doen.

### E. Volatiliteitsschaling — geen instap, maar de positiegrootte

**Dit is de enige optie op de lijst die geen nieuwe sectie is.**

Het opvallendste aan de TSMOM-discussie hierboven is waar de critici op
uitkomen: het rendement zit grotendeels in het schalen op volatiliteit, niet in
het signaal. Kleiner handelen als de markt hard beweegt, groter als hij rustig
is.

**Jarvis doet dat niet.** De inzet is 2% van de rekening, altijd, ongeacht het
regime. De stop past zich aan de ATR aan, maar de ingezette R niet aan de
volatiliteit van de periode.

**Het academische anker, en de afbraak ervan.** Moreira en Muir,
*Volatility-Managed Portfolios* (*Journal of Finance* 2017): minder risico nemen
als de volatiliteit hoog is levert grote alfa's op, over de markt, waarde,
momentum, winstgevendheid en de valuta-carry.

Maar dat is niet blijven staan, en dat hoort hier voordat je er een uur in
steekt:

- Cederburg en co-auteurs vinden dat het buiten de steekproef faalt.
- Barroso en Detzel vinden dat het handelskosten niet overleeft.
- DeMiguel en co-auteurs (*Journal of Finance* 2024) vinden dat
  volatiliteitsgestuurde portefeuilles hun ongestuurde tegenhangers niet
  systematisch verslaan, en dat redelijke out-of-sample-versies het juist
  slechter doen.

Dat is een van de grondiger afbraken in deze lijst. Ik had dit gisteravond als
"het meest kansrijke" neergezet op basis van de TSMOM-discussie; dat was te snel
en dit is de correctie.

**Voor jou, en dit is waarom het er tóch nog staat.** Nul extra trades, nul
extra kosten, nul nieuwe instapregels, geen enkele nieuwe kandidaat en dus geen
extra straf op de Bonferroni-lat. Het is dezelfde 3.464 trades, anders gewogen,
narekenbaar op de CSV's die er al liggen zonder één replay.

**Oordeel.** De literatuur zegt inmiddels grotendeels nee. Maar het is de
goedkoopste vraag in het hele document — een middag rekenen op bestaande data,
zonder dat er iets live kan gaan — en jouw situatie is niet die van een
aandelenfactor: jij vraagt niet of het alfa toevoegt, maar of je in rustige
periodes te klein en in wilde periodes te groot zit. Doe hem, maar verwacht nee.

### F. Intraday time-series momentum op Bitcoin

**Claim.** Binnen de dag zet Bitcoin zijn eigen richting door, en die
voorspelbaarheid is groot genoeg om na kosten over te houden.

**Bewijs.** *Bitcoin Intraday Time-Series Momentum* (University of Reading,
werkdocument). De auteurs stellen expliciet dat het effect **verhandelbaar** is
in de Bitcoinmarkt en dat timing op intraday-voorspellers meer oplevert dan
altijd-long of kopen-en-vasthouden. Daarnaast is er bredere literatuur over
time-series momentum in crypto.

**Mechanisme.** Zwakker dan bij de andere opties, en dat moet gezegd. Het
gebruikelijke verhaal is een markt met veel particuliere deelnemers, geen
sluitingstijd en trage informatieverwerking. Dat is een plausibel verhaal, geen
aangetoond mechanisme met een aanwijsbare verliezende tegenpartij.

**Voor jou.** BTCUSD staat al in je catalogus, en sectie vijftien handelt die
markt al. Let op: die sectie verdiende +€15,62 in 180 dagen terwijl de
spreadpoort in datzelfde venster 7.414 setups weigerde die samen **−2241,89 R**
waren. Op BTC is jouw kostenprobleem het allergrootst van alle markten die je
hebt.

**Oordeel.** Interessante claim, zwakste bewijskwaliteit van de lijst
(werkdocument, geen top-tijdschrift), en het staat op de markt waar jouw kosten
het hardst bijten. Onderaan de stapel.

### G. Wat er structureel ontbreekt, zonder dat ik er een paper bij heb

Niet als aanbeveling, wel als eerlijke kaart van het gat:

- **Iets dat langer dan een dag aanhoudt.** Je maximale horizon is intraday.
  Kosten zijn een vast bedrag per rondje; een trade die vijf dagen loopt heeft
  dezelfde spread over een veel grotere R. Dat is dezelfde som als in optie E,
  van de andere kant.
- **Twee markten tegen elkaar.** Goud tegen de dollar, NDX tegen SPX. Alles wat
  je hebt kijkt naar één grafiek tegelijk.
- **Handelen op een agenda in plaats van op een grafiek.** Optie B is hiervan
  het enige voorbeeld en het is er meteen het bewijs van dat de lage frequentie
  klemt.

### WAARSCHUWING BIJ ALLES HIERBOVEN: de meetbank rekent geen swap

Dit ondergraaft het advies in dit document zelf, dus het staat hier en niet in
een voetnoot.

`PositionSizer._cost_share` rekent drie dingen: commissie, slippage en spread.
Geen swap. Dat is vandaag geen fout — alles wat Jarvis handelt sluit intraday
en wordt om 20:50 UTC platgelegd, dus er wordt nooit een nacht doorgerold.

**Maar elk voorstel om langer vast te houden loopt hier tegenaan.** Swap is een
bedrag per nacht per lot. Een trade die vijf dagen loopt betaalt hem vijf keer,
en de bank die je gebruikt om te beoordelen of dat de moeite waard is, rekent
hem nul keer.

Dat is de gevaarlijkste soort fout die er is: hij wijst de optimistische kant
op, precies bij de strategie waar dit document naartoe duwt. Een meerdaagse
sectie zou hier prachtig meten en live tegenvallen, en dat is letterlijk het
patroon waar dit hele project aan lijdt.

**Het goede nieuws: het getal ligt er al.** `journal/database.py` bewaart een
`swap`-kolom per positie en `reporting/pdf_report.py` telt hem al bij de winst
op. Je eigen journaal weet dus wat Eightcap je rekent. Voordat er ook maar iets
gebouwd wordt dat een nacht overhoudt, moet die kolom in `_cost_share` — anders
meet je iets wat je niet kunt handelen.

---

## Wat ik hieruit zou doen, in deze volgorde

1. **De domme goud-sessietest (optie D).** Kost een uur, gebruikt data die er al
   ligt, en beantwoordt de vraag of je enige werkende sectie een model is of een
   klok. Geen nieuwe kandidaat, dus geen straf op de lat. **Dit is gebouwd en
   staat klaar: `domtest.cmd`.**
2. **Volatiliteitsschaling narekenen op de bestaande CSV's (optie E).** Ook geen
   nieuwe kandidaat, ook geen replay nodig — maar lees eerst wat er in optie E
   staat over Cederburg, Barroso-Detzel en DeMiguel. De literatuur zegt nee. Het
   is de moeite omdat het bijna niets kost, niet omdat het gaat lukken.
3. **Pas daarna, en hooguit één:** intraday momentum (optie A), vooraf
   vastgelegd in `docs/hypotheses/`, met het holdoutjaar dat je al hebt en dat
   precies één keer bekeken wordt.

## Wat ik niet zou doen

Twintig kandidaten tegelijk. Je eigen `_bonferroni_t` zegt waarom: bij 3.000
trades heeft één vooraf vastgelegde kandidaat +0,043 R per trade nodig, en dat
haalt sectie zes ruim. Bij duizend kandidaten is het +0,089 en haalt sectie zes
het in zijn slechte jaar niet meer.

Je hebt genoeg data om precies één ding te bewijzen. Dat is geen beperking om
omheen te werken; dat is de reden dat je het deze keer wél kunt bewijzen.

---

## Bronnen

- [McLean & Pontiff, *Does Academic Research Destroy Stock Return Predictability?*, Journal of Finance 2016](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)
- [Novy-Marx & Velikov, *A Taxonomy of Anomalies and Their Trading Costs*, Review of Financial Studies 2016](https://academic.oup.com/rfs/article-abstract/29/1/104/1844518)
- [Gao, Han, Li & Zhou, *Market Intraday Momentum*, Journal of Financial Economics 2018](https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351)
- [Lucca & Moench, *The Pre-FOMC Announcement Drift*, Journal of Finance 2015](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12196)
- [Boyarchenko, Larsen & Whelan, *The Overnight Drift*, NY Fed Staff Report 917](https://www.newyorkfed.org/research/staff_reports/sr917)
- [*The Disappearing Overnight Drift*, Liberty Street Economics, juli 2026](https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/)
- [Moskowitz, Ooi & Pedersen, *Time Series Momentum*, Journal of Financial Economics 2012](https://www.sciencedirect.com/science/article/pii/S0304405X11002613)
- [Lou, Polk & Skouras, *A Tug of War: Overnight versus Intraday Expected Returns*, JFE 2019](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650)
- [Iwatsubo, Watkins & Xu, *Intraday Seasonality in Efficiency, Liquidity, Volatility and Volume: Platinum and Gold Futures in Tokyo and New York*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3021533)
- [Moreira & Muir, *Volatility-Managed Portfolios*, Journal of Finance 2017](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513)
- [DeMiguel et al., *A Multifactor Perspective on Volatility-Managed Portfolios*, Journal of Finance 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13395)
- [*On the performance of volatility-managed portfolios*, Journal of Financial Economics](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X)
- [*Bitcoin Intraday Time-Series Momentum*, University of Reading](https://centaur.reading.ac.uk/100181/3/21Sep2021Bitcoin%20Intraday%20Time-Series%20Momentum.R2.pdf)
- [Barber, Lee, Liu & Odean, *Day Trading for a Living?* / *The Profitability of Day Traders* — Taiwan 1992-2006](https://www.researchgate.net/publication/228289199_The_Profitability_of_Day_Traders)
- [Chague, De-Losso & Giovannetti, *Day Trading for a Living?* — Brazilië 2013-2015](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101)
