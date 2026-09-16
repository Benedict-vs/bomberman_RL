# Adversarial audit: Mixed-kill position balance (2026-09-04)

## Auftrag und vorlaeufiger Status

Dieser Audit versucht ausdruecklich, die Behauptung zu widerlegen, dass die
konkrete eingefrorene Mixed-kill-Policy gegenueber dem konkreten eingefrorenen
`dqn_task3_seed13` im Feld aus drei `rule_based_agent` ueber alle vier
Listenpositionen besser ist. Agentencode und `BEN.md` bleiben unveraendert.

Der Bericht wird waehrend der unabhaengigen Rekonstruktion fortgeschrieben.
Der Rohdatenbefund ist inzwischen stabil; das Endurteil steht unten.

## Zuerst festgestellte Artefakte

- Mixed-kill-Modell auf Disk: SHA-256
  `d50ae3d1ce80018a8a834f7df28cff3dfead5c0c99d578d7af1604a3cf0806c6`.
- Incumbent-Modell auf Disk: SHA-256
  `1285ac5cb78a25b0e6cc6a0e0a68fdd86e153db537832938f940cb1ad4fd8b63`.
- Aktuelle Callbacks: Mixed-kill `4270ab...f4ef`, Incumbent
  `3a6017...11f` (vollstaendige Hashes werden unten geprueft).
- Fuer beide Policies existieren Slot-0-Dateien sowie je eine Datei fuer Slots
  1, 2 und 3. Fuer Slot 0 existieren zusaetzliche Wiederholungen; ihre Auswahl
  und Gewichtung ist ein zentraler Auditpunkt.

## Rohdaten, Auswahl und Provenienz

Ich habe die neun ausgewaehlten CSVs unabhaengig eingelesen (vier Mixed-kill,
fuenf Incumbent, weil dessen Slot 0 zweimal wiederholt wurde). Jede Datei hat
exakt 4.000 Zeilen, 1.000 Runden und die Seeds `20260731..20261730`. Pro Seed
existiert genau eine Zeile des Zielagenten; dessen `slot` stimmt in allen
Dateien mit der Agentenreihenfolge der Metadaten ueberein. Alle Lineups bestehen
aus genau der Zielpolicy und drei `rule_based_agent`.

Alle ausgewaehlten Runs haben `classic` und identische relevante Regeln:
17x17, 400 Schritte, Bombenstaerke 3, Timer 4, Explosionstimer 2, Timeout 0,5 s,
Coin +1, Kill +5, 9 Coins und Crate-Dichte 0,75. Die Mixed-kill-Runs wurden mit
denselben dokumentierten `BM_*`-Schaltern ausgefuehrt; beim Incumbent war neben
`BM_QUIET_LOGS=1` kein verhaltensaendernder Schalter gesetzt.

Die Modellhashes in jedem Metadatensatz stimmen mit den heute vorhandenen
Dateien ueberein:

- Mixed-kill: `d50ae3d1ce80018a8a834f7df28cff3dfead5c0c99d578d7af1604a3cf0806c6`;
- Incumbent: `1285ac5cb78a25b0e6cc6a0e0a68fdd86e153db537832938f940cb1ad4fd8b63`.

Auch die Callback-Hashes sind innerhalb jeder ausgewaehlten Policy konstant
und stimmen mit den aktuellen Dateien ueberein: Mixed-kill
`4270ab56cf95f358c9b67dfc404d7e437bcee97494bf93df09b935a371c2f4ef`,
Incumbent `3a601738fef20b2f6101a26bece2908a2b92193cd52c6298c04c1cd5b925d11f`.
Damit ist insbesondere die alte Mixed-kill-Slot-0-Datei mit Callback
`f50de2...17c0` zurecht aus dem primaeren Schaetzer ausgeschlossen.

Es bleibt eine echte Provenienzgrenze: `evaluate.py` hasht nur `callbacks.py`
und das Gewichtsfile, nicht die importierten `features.py` und `model.py`.
Ausserdem tragen die Runs `*-dirty`; der Incumbent-Code ist am angegebenen
Commit nicht vollstaendig rekonstruierbar. Die aktuellen Feature-/Modelldateien
sind zwischen den Agentordnern erwartungsgemaess nicht bytegleich. Fuer die
frischen Mixed-kill-Runs spricht die Dateizeitfolge dafuer, dass `features.py`
vor den Messungen unveraendert war, aber Metadaten beweisen das nicht. Diese
Luecke widerlegt die CSV-Ergebnisse nicht, verbietet jedoch die staerkere
Behauptung einer vollstaendig aus Git und Metadaten reproduzierbaren
Inference-Policy.

## Arithmetik und Todessemantik

Fuer jede Zielagentenzeile gilt exakt `score = coins + 5*kills`. Ebenfalls gilt
`died = suicides + killed_by_opponent` und `survived = 1-died`. Damit sind die
publizierten Score- und Death-Komponenten intern konsistent.

Der Name `killed_by_opponent` ist geringfuegig zu stark: Bei gleichzeitigem
Treffer durch eigene und fremde Explosion setzt das Framework `KILLED_SELF`;
`evaluate.py` klassifiziert den Tod dann vollstaendig als Suizid. Die Spalten
sind daher genauer „Tod mit eigener Explosion beteiligt“ versus „Tod ohne
eigene Explosion“. Das aendert weder Score noch Survival und kann den klaren
Survivalbefund nicht erzeugen; kausale Aussagen ueber den exakten Bombenbesitzer
sollten daraus aber nicht abgeleitet werden.

## Unabhaengige Rekonstruktion des Schaetzers

Wie in der frueheren Incumbent-Pruefung habe ich nicht 4.000 Slotbeobachtungen
als unabhaengige Stichprobe behandelt. Pro Policy und Arenaseed wird zuerst je
Slot gemittelt, dann werden vier Slots gleich gewichtet; erst die 1.000
Arenamittel werden verglichen. Beim Incumbent werden seine zwei Slot-0-Runs
innerhalb `(Seed, Slot)` gemittelt. Das ist kein Pseudoreplikationsfehler:
Slotgewichte bleiben 1/4 statt Slot 0 doppelt zu zaehlen.

Die unabhaengig rekonstruierten Primaerwerte sind:

| Slot | Mixed-kill | Incumbent | Differenz | Paired 95%-CI |
|---:|---:|---:|---:|---:|
| 0 | 3,761 | 3,3285 | +0,4325 | [+0,256; +0,614] |
| 1 | 3,876 | 3,351 | +0,525 | [+0,314; +0,739] |
| 2 | 3,781 | 3,280 | +0,501 | [+0,299; +0,705] |
| 3 | 3,810 | 3,372 | +0,438 | [+0,223; +0,651] |
| gleichgewichtet | 3,807 | 3,332875 | **+0,474125** | **[+0,373; +0,576]** |

Ein 100.000-Ziehungen-Signflip-Test hatte fuer den Gesamtschaetzer keinen
Treffer (Monte-Carlo-Obergrenze ungefaehr `1e-5`, nicht mathematisch `p=0`).
Zwanzig Bootstrap-Seeds ergaben stets klar positive Untergrenzen
`0,371..0,376`; der Befund ist nicht fragil. Die sehr enge Uebereinstimmung mit
der Primaeranalyse ist eine echte unabhaengige Rekonstruktion aus den CSVs.

## Versuche, den Effekt durch Analyseentscheidungen zu brechen

- **Asymmetrischer Slot-0-Aufwand.** Mixed-kill hat einen aktuellen Slot-0-Run,
  der Incumbent zwei. Wird statt ihres Mittels jeweils nur eine Incumbent-
  Wiederholung verwendet, ist der Gesamteffekt `+0,494` bzw. `+0,454`; beide
  CIs bleiben klar positiv. Mehr Incumbent-Wiederholungen reduzieren lediglich
  dessen Messrauschen und verzerren das Slotgewicht nicht.
- **Ein Slot treibt alles.** Leave-one-slot-out liefert `+0,457` bis `+0,488`;
  jede Untergrenze liegt mindestens bei `+0,345`. Kein Slot ist notwendig und
  keiner kehrt die Richtung um. Die Standardabweichung der vier Sloteffekte ist
  nur `0,046`.
- **Seedabhaengigkeit.** Die Korrelationen der seedweisen Differenzen zwischen
  Slots liegen nur zwischen -0,037 und +0,040. Moving-block-Bootstraps mit
  Blocklaengen 5, 10, 25 und 50 ergeben ebenfalls ausschliesslich positive
  Intervalle; die kleinste Untergrenze ist `+0,366`.
- **Komponentencheck.** Der Scoregewinn zerfaellt korrekt in Coins `+0,2535`
  und fuenfmal Kills `5*0,044125`, zusammen `+0,474125`. Zugleich fallen
  Suizide um `0,102`, Tode ohne eigene Explosion um `0,03425`, und Survival
  steigt um `0,13625`; alle Intervalle liegen klar auf der guenstigen Seite.
  Die konkrete Policy zeigt in diesem Feld also keinen versteckten
  Safety-Trade-off gegenueber diesem konkreten Incumbent (wohl aber gegenueber
  ihrem frueheren, bewusst sicheren Trainingskontrollarm).
- **Post-selection.** Mixed-kill wurde urspruenglich unter vielen Task-4-Armen
  als Kandidat gefunden; die alte Slot-0-Beobachtung ist daher selektionsnah.
  Die drei danach erhobenen, komplett positiven Slot-1/2/3-Runs sind aber eine
  starke frische Bestaetigung. Selbst der Ausschluss von Slot 0 ergibt
  `+0,488 [+0,368;+0,609]`. Winner's curse kann die enge Behauptung daher nicht
  erklaeren.

## Gegner-RNG und inferenzstatistische Grenze

`evaluate.py` setzt pro Runde `world.rng` und `np.random`, der
`rule_based_agent` verwendet aber zusaetzlich `random.shuffle` aus der
Python-Standardbibliothek. Diese RNG wird nicht pro Seed gesetzt. Die Runs der
beiden Policies teilen daher Arenen und Listenpositionen, aber **keine gepaarten
Gegnertrajektorien**. Der paired Bootstrap entfernt Arenavarianz, nicht
Gegnerrauschen. Das ist dieselbe Grenze wie im Incumbent-Audit und bedeutet:
Der Effekt ist ein Unterschied ueber die konkret gezogenen Gegner-RNG-Stroeme,
nicht deterministische seedweise Dominanz.

Der bekannte ungefaehre `0,12`-Score-Noise-Floor aus gleichen 1.000-Runden-
Wiederholungen ist deutlich kleiner als `+0,474`. Vier konsistent positive
Slots, Leave-one-slot-out und Blockbootstrap machen eine reine Gegner-RNG-
Erklaerung unplausibel. Formal ist der Signflip-Test hier zudem kein echtes
Randomisierungsdesign (Policyzuweisung wurde nicht randomisiert); der
Bootstrap ueber Arenacluster ist die wichtigere Unsicherheitsanalyse. Eine
weitere Wiederholung mit explizit gesetzter stdlib-RNG oder ein gemeinsamer
randomisierter A/B-Harness waere fuer eine strengere kausale Aussage besser.

## Laufzeit

Kein ausgewaehlter Run hat einen Schritt ueber 500 ms. Das globale Maximum ist
46,7317 ms fuer Mixed-kill und 13,8512 ms fuer den Incumbent. Mixed-kill hat
damit etwa Faktor 10,7 Reserve zur offiziellen Grenze. Das ist auf der
Entwicklungsmaschine klar sicher, aber noch kein Benchmark auf dem langsameren
Referenzprozessor; `think_max_ms` sollte in der finalen Docker-/CPU-Pruefung
weiter beobachtet werden.

## Kritik am Primaerskript

`scratchpad/ben_task4_mixed_kill_position_balance.py` implementiert die
entscheidende Clusterung und Slotgewichtung korrekt. Seine Assertions pruefen
Modellhash, Seedmenge, Rundenzahl, Zielslot, Szenario und Timeouts. Der
Begleittext beansprucht jedoch mehr als das Skript selbst prueft:

- es assertiert weder Callback-Hashes noch identische Settings oder Lineups;
- es prueft nicht `score = coins + 5*kills` oder die Death-Arithmetik;
- es hasht keine importierten Feature-/Modelldateien;
- `signflip_p` kann `0` ausgeben statt der ueblichen Monte-Carlo-Korrektur;
- die Fragilitaetsfunktion behandelt bei wechselnden Bootstrap-Urteilen
  `significant=False`, was konzeptionell unsauber ist, fuer diesen weit von null
  liegenden Befund aber folgenlos bleibt.

Diese Punkte habe ich direkt an Rohdaten und Metadaten nachgeprueft. Nur die
unvollstaendige Codeprovenienz bleibt als reale Einschraenkung bestehen.

## Endurteil

**Eingeschraenkt aber tragfaehig.** Ich konnte den positionsrobusten
Scorevorteil nicht widerlegen. Die engste zulaessige Behauptung lautet:

> In den vorhandenen 1.000 Standardarenen und den vier gleich gewichteten
> Listenpositionen erzielte die konkret gehashte Mixed-kill-Seed-11-Policy in
> separaten Spielen gegen drei `rule_based_agent` im Mittel 3,807 Punkte, der
> konkret gehashte Task-3-Seed-13-Incumbent 3,333; der arenastratifizierte
> Schaetzer betraegt +0,474 [ungefaehr +0,373; +0,576], bleibt in jedem Slot und
> nach Ausschluss jedes einzelnen Slots positiv und zeigt keine
> Laufzeitueberschreitung.

Nicht gestuetzt sind daraus: eine reproduzierbare Mixed-kill-Trainingsfamilie,
deterministische Dominanz bei identischen Gegnertrajektorien, Uebertragbarkeit
auf externe Gegner oder vollstaendige Reproduzierbarkeit der historischen
Inference-Software aus Git/Metadaten.

**Konkreter naechster Schritt:** den eingefrorenen Hash ohne weitere
Modellauswahl gegen das bereits festgelegte externe Trio pruefen, zuerst den
Quick100-Liveness-/Timing-Gate und danach bei bestandenem Gate Full1000 in
fester Reihenfolge. Fuer die spaetere finale Bestaetigung sollte der Evaluator
zusaetzlich `random.seed(base_seed + round_index)` setzen und Feature-/Model-
Quellhashes in die Metadaten aufnehmen; bestehende Vergleichsserien duerfen
dabei nicht still mit dem geaenderten RNG-Protokoll vermischt werden.

Rekonstruktionscode:
`scratchpad/audit_ben_task4_mixed_kill_position_balance/reconstruct.py`.

## External Quick100 Go/No-Go

### Adversariale Integritaetspruefung

Geprueft wurden genau:

- `ben_dqn_task3_seed13__task4_external_top3_quick100.{csv,meta.json}`;
- `ben_dqn_task4_mixed_kill_v1_2000ep_seed11__task4_external_top3_quick100.{csv,meta.json}`.

Beide Dateien sind vollstaendig: je 400 Zeilen, 100 eindeutige Runden und
Seeds `20260731..20260830`, mit vier eindeutigen Agenten pro Runde. Beide
Zielpolicies stehen in Slot 0; das externe Trio und seine Reihenfolge sind in
beiden Runs identisch: Li-Jesse Slot 1, Bindist Slot 2, Binary Slot 3. Szenario,
Regeln, Python/Plattform und Commitmarker stimmen ueberein.

Die Modell- und Callback-Provenienz der beiden Zielpolicies stimmt sowohl mit
dem positionsbalancierten Audit als auch mit den aktuellen Dateien ueberein:

- Incumbent-Modell `1285ac...8b63`, Callback `3a6017...11f`;
- Mixed-kill-Modell `d50ae3...06c6`, Callback `4270ab...f4ef` sowie die korrekt
  protokollierten Schalter `mixed_kill_v1`, 2.000 Episoden, Seed 11,
  `MODEL_VARIANT=trained`.

Alle 800 Zeilen bestehen die Score- und Death-Arithmetik. Es gab in keinem
Agenten einen Timeout. Fuer die Zielpolicies betragen die globalen Maxima nur
22,80 ms (Incumbent) und 23,50 ms (Mixed-kill). Auch die externen Agenten
blieben unter 500 ms; das hoechste Einzelmaximum war 223,85 ms bei Bindist.
Damit gibt es keinen Liveness-, Packaging- oder unmittelbaren Timing-Grund,
Full1000 abzubrechen. Der langsame Gesamtdurchsatz (etwa 7,6 bzw. 8,1 Minuten
fuer 100 Runden) prognostiziert allerdings rund 76--81 Minuten pro Full1000.

### Leistungssignal und Futility

Nur als Screen, nicht als Endergebnis: Mixed-kill erreicht Score 2,40, Kills
0,05, Suizide 0,39 und Survival 0,45; der Incumbent 2,11, 0,02, 0,42 und 0,36.
Die Primaerrichtung und alle groben Safety-Richtungen sprechen damit gegen
einen Futility-Stopp. Bei nur 100 Runden, schwerem heterogenem Feld und
ungepaarten Gegnerzufallsstroemen duerfen diese Abstaende weder als bewiesen
noch als Praezisionsschaetzung behandelt werden. Es war fuer diesen externen
Liveness-/Timing-Gate keine nachtraegliche CI-Futility-Schwelle vorab
festgelegt; eine solche jetzt aus dem Quick100 zu erfinden waere optional
stopping.

Wie zuvor bleibt die Gegner-RNG eine Einschraenkung: getrennte Evaluationslaeufe
teilen Arenaseeds, aber nicht notwendig die Zufallsentscheidungen der externen
Agenten. Ausserdem dokumentiert `evaluate.py` fuer die externen Wrapper nur
Callback-Hashes und keine dahinterliegenden Modellhashes. Die identischen
externen Callback-Hashes und Code-Namen in beiden am selben Stand erzeugten
Runs reichen fuer den geplanten Fixed-lineup-Vergleich, aber nicht fuer eine
universell reproduzierbare externe Rangliste. Die Full1000-Auswertung muss
diese Teilpaarung ausdruecklich nennen und darf externe Gegnerzeilen nicht als
zusaetzliche unabhaengige Replikationen zaehlen.

### Entscheidung

**GO fuer genau die zwei vorab geplanten Full1000-Laeufe.** Ich finde keinen
integren No-Go-Grund: richtige eingefrorene Zielmodelle, aktuelle Callbacks,
exaktes gemeinsames Lineup und Reihenfolge, vollstaendige Seedfolge, null
Overruns und kein Futility-Signal. Quick100 erfuellt damit ausschliesslich
seinen vorgesehenen Zweck als Liveness-/Timing-Gate. Es rechtfertigt noch keine
Promotion und keine Aussage, Mixed-kill sei extern besser.

Die beiden Full1000 muessen unveraendert mit denselben Slots, Seeds, Modellen,
Callbacks und Gegnern laufen; beide Ergebnisse bleiben unabhaengig vom Ausgang
erhalten. Primaer ist der Zielagenten-Score pro Arena. Wegen nicht gemeinsam
kontrollierter Gegnertrajektorien ist ein arenastratifizierter Vergleich nur
teilgepaart und muss zusammen mit absoluten Mitteln, CIs und dem bekannten
Gegner-RNG-Noise-Floor berichtet werden.
