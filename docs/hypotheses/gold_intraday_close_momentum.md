# Goud: intraday momentum naar de COMEX-close

## Vooraf vastgelegde hypothese

Baltussen, Da, Lammers en Martens (JFE 2021) vinden bij meer dan zestig futures
dat het rendement vanaf de vorige marktclose tot dertig minuten voor de huidige
close dezelfde richting voorspelt in de laatste dertig minuten. Voor COMEX GC
gebruiken zij 08:20--13:30 New York-tijd. Hun economische verklaring is
uitgestelde deltahedging door partijen met short-gamma blootstelling.

Deze test vertaalt precies die ene regel naar XAUUSD:

1. vorige referentieclose: laatste M5-close op of voor 13:30 New York-tijd;
2. signaal: prijs om 13:00 minus die vorige referentieclose;
3. positief signaal = long, negatief signaal = short;
4. entry op de 13:00 M5-open en exit op de 13:30 M5-open;
5. maximaal een trade per geldige handelsdag;
6. New York-tijd wordt via `America/New_York` omgerekend, dus DST is geen
   handmatige zomer/winterparameter;
7. resultaat in R gebruikt vooraf 0,8 x M5 ATR(14), met 0,066 R kosten per trade.

Er is geen magnitude-drempel, weekday-filter, nieuwsfilter, exit-grid of
uursweep. Die zouden van een replicatie een optimalisatie maken.

## Wie staat vermoedelijk aan de andere kant?

Partijen die hun deltahedge pas rond de settlement bijwerken. Bij een stijgende
dag moeten short-gamma hedgers bijkopen en bij een dalende dag verkopen. Als dit
mechanisme nog bestaat, creëert die gedwongen stroom momentum in het laatste
halfuur.

## Falsificatie en rapportage

Rapporteer bruto en netto R, winrate, gemiddelde R, t-statistiek, maximum
drawdown en elk kalenderjaar afzonderlijk. Vergelijk daarnaast met altijd long
in hetzelfde halfuur. De regel is verworpen wanneer netto expectancy niet
positief is, wanneer het teken niet in beide beschikbare onafhankelijke jaren
positief blijft, of wanneer het resultaat hoofdzakelijk door één jaar wordt
gedragen. Een positieve verkenningsrun geeft geen toestemming voor live.

Bron: https://doi.org/10.1016/j.jfineco.2021.04.029

