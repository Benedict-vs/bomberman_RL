# Ben Task 4: Was die Verbesserung derzeit begrenzt

Datum: 2026-09-13. Auftrag: Bens eigenen Entwicklungszweig ansehen und nächste
Verbesserung begründen. Gelesen: `ben_coin_collector`, `ben_coin_collector_dqn`,
`ben_task2`, dessen Evaluationswrapper, `ben_task3`, Schwerpunkt `ben_task4`,
zugehörige Werkzeuge, BEN.md/CODEX.md und vorhandene Ergebnisse. Keine Analyse
der Implementierungen von Benedict, Maxi oder externen Agenten. Keine Spiele,
Trainings, Installationen oder Änderungen an Agenten/Modellen gestartet.

## 1. Ausgangslage: Punkte fehlen an unterschiedlichen Stellen

Vorhandene gemeinsame External-Trio-Full1000-Messung:
`results/eval/task4_tournament/ben_dqn_task4_mixed_kill_v1_2000ep_seed11__task4_external_top3_eval1000.csv`.
Erneut mit dem vorhandenen `tools/analyze.py --preset task4` ausgewertet.

| Agent | Score | Coins | Kills | Suizide |
|---|---:|---:|---:|---:|
| ben_task4 Mixed-Kill Seed 11 | 2,192 | 1,902 | 0,058 | 42,9 % |
| Li-Jesse | 3,675 | 2,880 | 0,159 | 21,9 % |
| Bindist | 3,129 | 1,904 | 0,245 | 37,8 % |
| Binary | 3,224 | 2,109 | 0,223 | 47,6 % |

Coins sind aus dem auditierten offiziellen Zusammenhang `score = coins + 5*kills`
rekonstruiert. Die Tabelle beschreibt diese konkrete Messung mit fester
Agentenreihenfolge, keine allgemein bewiesene Rangfolge aller Slots/Trainingsseeds.

Bindists Scorevorsprung von 0,937 besteht aus 0,002 Coins und 0,935 Killpunkten;
Li-Jesses Vorsprung von 1,483 aus 0,978 Coins und 0,505 Killpunkten. Binary zeigt,
dass geringere Suizidrate allein keinen höheren Score garantiert. Bens 429
Suizide gegenüber 127 gegnerverursachten Toden machen die eigene Bombensicherheit
trotzdem zu einem relevanten Ansatzpunkt. Daraus folgt noch nicht, welcher
einzelne Fehler die Todesfälle verursacht.

Die CNN-Grundstruktur mit zwei räumlichen Verkleinerungen und einem Q-Kopf wurde
von Task 2 über Task 3 übernommen. Dueling und der Hilfskopf ändern diese
Grundlage nur begrenzt. Die Rohkanäle enthalten Bombentimer, Gefahren, Gegner
und eigene Position; es wäre falsch, dem Netz jede zeitliche Information oder
die prinzipielle Fähigkeit zum Lernen einer Flucht abzusprechen.

## 2. Bestätigte Konfigurationsfalle: Standardstart lädt die Baseline

`agent_code/ben_task4/callbacks.py:15–16,28–31,191–211` kombiniert standardmäßig
`mixed_kill_v1`, 5.000 Episoden und `MODEL_VARIANT=auto`. Der gesuchte
`ben_task4_mixed_kill_v1_5000ep_seed11.pt` existiert hier nicht; die Auswahl fällt
auf `ben_task4_baseline_v1_5000ep_seed11.pt` zurück. Ein reiner Modulimport
hat genau diese Auswahl bestätigt, ohne eine Policy auszuführen.

Der ausgewählte Incumbent heißt dagegen
`ben_task4_mixed_kill_v1_2000ep_seed11.pt`. Explizite Befehle mit Arm
`mixed_kill_v1`, Episoden `2000`, Seed `11`, Variante `trained` vermeiden
die Falle. Die benannten historischen Metadaten dürfen einzeln geprüft werden;
daraus folgt keine pauschale Verwechslung aller bisherigen Experimente.

Empfehlung: Ein expliziter Inferenzstandard für den Incumbent und ein sichtbarer
Modellpfad/Hash bei Start, unabhängig von der gewünschten neuen Trainingsdauer.
Bei ausdrücklich angefordertem Modell keine stille Ersatzwahl. Noch nicht
implementiert; dies verbessert Zuverlässigkeit, nicht die gelernte Policy.

## 3. Bestätigte Grenze der Fluchtinformation

`features.py:422–470`: `_reachable_escape_tiles` sucht geometrisch bis zu vier
Schritte weit. Es prüft Gefahr nur am Ziel, nicht unterwegs. Außerdem liefert
es bei `bomb_available=False` ausschließlich Nullen; es wurde als Information
vor einer hypothetischen neuen Bombe gebaut, nicht als vollständiger Fluchtplan
für die bereits liegende eigene Bombe.

Reproduktion in `scratchpad/ben_task4_diagnostic_probes_20260913.py`:
Start (1,1), einziger Ausgang (2,1) mit aktiver Explosion, danach ein Gang um
die Ecke nach (3,2). Das Feature markiert (3,2) trotzdem als erreichbar. Eine
spätere Kachel außerhalb der Explosion macht einen aktuell tödlichen Weg nicht
sicher. Dies ist ein synthetischer Nachweis der Featuregrenze, keine Messung
ihrer Häufigkeit in unseren verlorenen Partien.

Auch der verworfene Zusatzkanal `_temporal_safety_slack` (`features.py:317–378`)
ist kein vollständiges zeitabhängiges Erreichbarkeitsmodell: eine früheste
Gefahrenzeit pro Kachel, nur eine Besuchszeit, keine WAIT-Kante, kein späteres
Wiederbetreten nach abgeklungener Explosion. Er modelliert außerdem keine neu
gelegte hypothetische eigene Bombe. Sein negatives Ergebnis widerlegt eine
vollständigere Zeitdarstellung daher nicht.

Empfohlene neue Lernhypothese: Zustandskarten für räumlich UND zeitlich
erreichbare sichere Bereiche beziehungsweise Explosionen über die nächsten
Zeitschritte helfen dem Netz bei eigener Bombenflucht und bei Annäherung zum
Angriff. Warten, Explosionsdauer, Bombenblockierung, dynamische Kisten und
die tatsächliche Update-Reihenfolge müssen dabei vor einem Pilot separat
geprüft werden. Gegnerzukünfte bleiben unbekannt; eine statische Belegung darf
nicht als garantierte Zukunft ausgegeben werden.

Der Input soll räumliche Zustandsinformation liefern. Keine vorgeschriebene
beste Aktion, keine kopierte Gegnerpolicy und keine automatische taktische
Aktionswahl als Ersatz für das Netz. Bestehende Kanäle zunächst erhalten;
neue Kanäle bei Kontrolle null und bei Kandidat informativ, identische
initiale Gewichte mit auf null gesetzten zusätzlichen Eingangsgewichten.
Ob dies den offiziellen Score verbessert, bleibt eine offene Hypothese.

## 4. Reproduzierte Inkonsistenz am überlebten Rundenende

`train.py:437–499`, `train.py:682–693` und
`replay_buffer.py:152–168`: Beim Tod wird der Folgezustand für das
Safety-Potential als terminal mit Phi=0 behandelt. Bei überlebtem Rundenende
bleibt der zuvor berechnete Reward erhalten und nur das `done`-Flag wird gesetzt.

Der Diagnosecode führt den tatsächlichen Überlebenszweig mit gemocktem
`torch.save` aus, ohne Optimierung. Für einen sicheren Übergang ohne Ereignisse
steht im terminalen Replay `−0,06`; nach der bei Tod verwendeten Konvention
`Phi(terminal)=0` wären es `−1,05`. Der Restterm ist `+0,99`.

Dies ist eine Inkonsistenz zur beabsichtigten rein potentialbasierten Formung,
kein fehlender Todesreward und kein Beweis für die Hauptursache der Niederlagen.
Ein Fix muss echte Rundenenden/Timeout-Semantik und N-step berücksichtigen und
ist als eigene Lernänderung zu behandeln. Einfach den positiven Restterm zu
entfernen könnte den bisherigen Überlebensanreiz verändern; ein besserer Score
ist deshalb nicht garantiert. Die gleiche strukturelle Stelle existiert bereits
in Bens Task-2- und Task-3-Trainingscode; historische Modelle nicht umschreiben.

## 5. Plateau- und Fortsetzungsurteile waren zu weitreichend

Unabhängige zweite Codeprüfung:
`scratchpad/audit_ben_task4_training_method_second_review_20260913.md`.

- `plateau_check.py` ist eine Heuristik: zwei aufeinanderfolgende Änderungen
  kleiner als 0,10, ohne Konfidenzintervall oder Äquivalenztest. Sie kann auch
  Rückgang als Plateau und eine Erholung unter dem früheren Niveau als
  `still_improving` bezeichnen. Safety wird ausgegeben, steuert aber den
  Exitstatus nicht. Der Pilot behandelt Analysefehler nicht als Abbruch.
- Neu berechneter gepaarter Vergleich der bereits vorhandenen Kontroll-
  Checkpoints 500→1000: Score `+0,020 [−0,410;+0,470]`. Diese Daten begrenzen
  einen möglichen Gewinn nicht auf weniger als 0,10. Korrekt ist: kein Gewinn
  nachgewiesen; ein enges Plateau ist ebenfalls nicht nachgewiesen.
- Checkpoints speichern nur Online-Netzgewichte. Neustarts setzen Adam,
  Replay, Target-Netz und Schrittzähler zurück. Der 5.000-Übergänge-Warm-up
  beginnt erneut; beim Phasenpilot bleibt epsilon zwar bei 0,05, aber der
  Lernzustand wird nicht fortgesetzt. Vier 250er-Prozesse sind keine
  unterbrechungsfreie 1.000er-Fortsetzung. Beide Arme können trotzdem fair
  unter genau diesem Neustartprotokoll verglichen werden.
- Der separate Lernraten-Checkpointvergleich stammt hingegen aus je einem
  durchgängigen Lauf; ihm darf man keine Neustarts zwischen den evaluierten
  Checkpoints unterstellen.
- Signifikant schlechtere einzelne Full1000-Endmodelle bleiben negative
  Ergebnisse. Sie beweisen keine allgemeine Leistungsgrenze des DQN.
- Quick100 mit einer Zusatzbedingung `Kills müssen nominell steigen` kann
  aussichtsreiche Kandidaten verwerfen: eine Differenz von −0,01 entspricht
  einem Kill im Sample. Maßgeblich bleibt Score; Sicherheitsgrenzen und
  Stichprobengröße müssen vorab festgelegt und Unsicherheit berücksichtigt
  werden. Eine fehlende Signifikanz beweist weder Gleichwertigkeit noch Nutzen.

## 6. Korrektur zur letzten Mehrseed-Entscheidung

Seed 12 gegen Seed 13 allein beantwortet nicht, ob einer den Incumbent schlägt.
Die pauschale Schlussfolgerung aus diesem Vergleich war unzureichend begründet.
Ergänzende Auswertung der vorhandenen, älteren Seed-11-Quick100-Dateien gegen
Seed 12: External-Score `+0,140 [−0,290;+0,580]`, 3RB-Score
`−1,220 [−1,890;−0,570]`, 3RB-Kills `−0,170 [−0,280;−0,060]`.
Diese historischen Vergleiche unterstützen Zurückhaltung bei Seed 12 und
geben keinen Anlass zur Promotion. Sie sind keine frische synchronisierte
Bestätigung und trennen Trainings- und Gegnerzufall nicht vollständig.

## 7. Konkrete Reihenfolge der nächsten Arbeit

1. Modellwahl und tatsächlichen Lernzustand eindeutig machen: Inferenzmodell
   explizit, echte Fortsetzungsdateien getrennt von reinen Inferenzgewichten,
   Quelle/Hyperparameter/Schrittzähler nachvollziehbar protokollieren. Ein
   Fortsetzungstest vergleicht zwei ununterbrochene Updates mit einem gespeicherten
   und wiederaufgenommenen Ablauf auf identischen Übergängen; kein Langtraining
   nötig, um die Semantik zu prüfen.
2. Terminal-Shaping separat korrekt spezifizieren, reproduzierten Fall als
   Regressionstest übernehmen und später als eigenen kontrollierten Pilot
   behandeln. Nicht still mit dem neuen Feature kombinieren.
3. Priorisierte neue Verhaltenshypothese: zeitabhängige Sicherheitskarten.
   Erst künstliche Engpässe, aktive/verzögert auslaufende Explosionen und
   Bombenflucht testen; dann ein isolierter Kontroll-/Kandidatenversuch vom
   selben expliziten Incumbent, auf identischer Spielerverteilung und mit
   durchgängigem Training. Keine gleichzeitige Änderung von Rewards, Replay-
   Sampling und Architektur. Bereits vorhandene Roh-Timer bleiben sichtbar.
4. Entwicklung weiter an festen Arenaseeds 20260731 prüfen. Quick100 dient
   zur Orientierung; aussichtsreiche oder unklare Fälle bei vorab festgelegtem
   Budget auf 300/1000 Runden ausweiten. Eine Erweiterung desselben Präfixes
   ist keine unabhängige Replikation. Slotbalance, mehrere Trainingsseeds und
   Gegnerzufall für Bestätigung berücksichtigen. Score primär, Coins/Kills
   erklären ihn, Suizide als vorregistrierte Sicherheitsgrenze.
5. Dauer nur unter klarer Fortsetzungssemantik beurteilen: z. B. Checkpoints
   0/250/500/1000/2000 innerhalb eines Laufs. Automatische Kontrolle soll
   Fortschritt, Unsicherheit, Regression und ausreichend eingegrenztes Plateau
   unterscheiden. Weitere Dauer erst anhand dieses Verlaufs begründen.

Keine neuen Trainingsbefehle oder Produktionsänderungen in diesem Review.
Es gibt konkrete, überprüfbare Verbesserungsansätze; ein Sieg über das externe
Trio lässt sich daraus noch nicht versprechen.
