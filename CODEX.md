# Bomberman RL — Codex Übergabestand

Stand: 2026-09-13  
Projekt: `/Users/bengschwend/Projects/bomberman_RL`  
Besitzer/Entscheider: Ben

Diese Datei ist die kompakte Übergabe für neue Codex-Chats. Zuerst diese Datei
lesen, danach nur bei Bedarf `BEN.md`, `AGENTS.md`, `KONZEPT.md` oder die genannten
Audit-Dateien öffnen.

## Aktuelle Neubewertung vom 13.09. — vor älteren Abschlussplänen lesen

Ben möchte `ben_task4` weiter verbessern. Der neue Review ist
`scratchpad/audit_ben_task4_improvement_review_20260913.md`; unabhängige
Methodenprüfung: `scratchpad/audit_ben_task4_training_method_second_review_20260913.md`.
Ältere pauschale Aussagen weiter unten über nutzloses längeres Training oder
einen endgültigen Entwicklungsabschluss sind durch diese Neubewertung begrenzt:

- Die Plateau-Kontrolle ist eine Heuristik ohne Unsicherheitsprüfung.
  Kontrollcheckpoint 500→1000: Score `+0,020 [−0,410;+0,470]`;
  damit weder Gewinn noch enges Plateau nachgewiesen.
- Fortsetzungen laden nur Online-Gewichte; Replay, Adam, Target-Netz und
  Zähler starten neu. Eine echte Trainingsfortsetzung ist noch nicht implementiert.
- `ben_task4` ohne explizite Variablen lädt wegen `5000`/`auto` die ältere
  Baseline. Für den Incumbent ausdrücklich `mixed_kill_v1`, `2000`, Seed `11`,
  Variante `trained` verwenden. Inferenzstandard noch nicht repariert.
- Escape-Kanal prüft keine Gefahr auf Zwischenkacheln und ist bei belegter
  eigener Bombe null. Neuer Ansatz: vollständigere zeitabhängige Zustandskarten,
  keine vorgegebene beste Aktion. Noch nicht implementiert oder trainiert.
- Am überlebten Rundenende bleibt ein Safety-Potential-Restterm erhalten;
  reproduziert mit `−0,06` statt `−1,05` bei sicherem Terminalzustand.
  Separater Korrekturpilot erforderlich, nicht mit neuem Feature vermischen.

Nächste Arbeit: Modellwahl/Fortsetzungssemantik und Trainingskontrolle eindeutig
machen, Terminal-Shaping separat behandeln und anschließend einen kontrollierten
Zeit-Sicherheitsfeature-Pilot vorbereiten. Score primär; Quick100 kann kleine
Gewinne nicht zuverlässig ausschließen. Der auditiert ausgewählte Seed-11-
Incumbent bleibt erhalten. Diagnoseproben:
`scratchpad/ben_task4_diagnostic_probes_20260913.py` (kein Spiel/Training).

### Fortsetzungszustand ist jetzt implementiert

`ben_task4` kann eine echte Trainingsfortsetzung verwenden. Mit
`BM_TASK4_SAVE_TRAINING_STATE=1` speichert jeder 100-Episoden-Checkpoint einen
separaten, atomaren Trainingszustand unter
`results/train/ben_task4/task4_<arm>_<total>ep_seed<seed>__training_state.pt`.
Er enthält Online-/Target-Netz, Adam, Replay, N-step-Queue, Zähler und
Zufallszustände. Standardmäßig ist das Schreiben deaktiviert, weil ein voller
Replay-Snapshot groß werden kann. Resume verlangt einen vertrauenswürdigen,
relativen `BM_TASK4_RESUME_STATE_FILE`; Arm, Seed, Eingabekanäle, Architektur
und Replay-Konfiguration werden geprüft. Lokale Runden zählen danach mit einem
Offset weiter. Test: `agent_code/ben_task4/test_training_resume.py`; insgesamt
37 Task-4-Tests bestanden. Dokumentation: BEN.md, Eintrag 13.09.

## Verbindliche Zusammenarbeit

- Ben startet Trainings und Evaluationen selbst. Codex bereitet Befehle vor, prüft
  Artefakte und analysiert Ergebnisse, startet aber keine Trainings- oder
  Evaluationsläufe eigenständig.
- Keine Installation anfordern oder durchführen. Docker ist auf Bens Mac nicht
  verfügbar; Docker-Schritte nicht weiterverfolgen.
- Jede neue Änderung, Entscheidung, Messung und jedes neue Experiment als neuen
  Eintrag oben in `BEN.md` dokumentieren. Alte Einträge nicht umschreiben.
- Nie `git add`, `git commit`, `git push` oder sonstiges Staging ausführen.
- Bei jeder Evaluation Fortschritt live zeigen: `tools/evaluate.py` ohne `--quiet`
  ausführen. `BM_QUIET_LOGS=1` darf gesetzt bleiben, weil es nur Agentenlogs betrifft.
- Vor längeren Läufen: Kontrollarm, Quelle, Seed, Gegnerfeld, Artefaktpfade und
  Advancement-/Sicherheits-Gates festlegen und adversarial auditieren.
- Zwei verschiedene Trainingsarme dürfen parallel von Ben gestartet werden, wenn
  ihre Artefaktpfade getrennt sind. Die gemeinsamen Framework-Agentenlogs können
  sich dabei überschreiben; für Provenienz und Verlauf zählen daher die getrennten
  Trainings-CSV und `.meta.json`. Nicht denselben Arm oder dieselben Zielartefakte
  parallel starten.

## Projektziel und Messregeln

Wir entwickeln einen lernenden Bomberman-Agenten für Task 4: `classic` mit Kisten
und drei Gegnern. Primär zählt der offizielle Gesamtscore, nicht die Winrate.
Wichtige Sekundärmetriken: `kills`, `suicides`, `killed_by_opponent`, `survived`
und `think_max_ms`. Tournament-Limit: 500 ms pro Schritt; der Agent läuft CPU-only.

Vergleiche müssen denselben Base-Seed `20260731` verwenden. Quick100 ist nur ein
Liveness-/Futility-Gate. Reportable Ergebnisse benötigen gewöhnlich Full1000 und
95-%-CI ohne Fragilitätsmarkierung. Ab Task 4 immer mehrere Trainingsseeds oder
eine klar begrenzte Aussage über den konkreten Checkpoint verwenden. Gegner-RNG ist
nicht vollständig reproduzierbar; gleiche Mittelwerte dürfen nicht vorausgesetzt
werden.

## Aktueller bester Stand

Der aktuelle Incumbent ist der konkrete auditiert ausgewählte Mixed-Kill-Checkpoint:

- Modell: `agent_code/ben_task4/ben_task4_mixed_kill_v1_2000ep_seed11.pt`
- SHA-256: `d50ae3d1ce80018a8a834f7df28cff3dfead5c0c99d578d7af1604a3cf0806c6`
- Training: 2.000 Episoden, Seed 11, Feld
  `peaceful_agent,rule_based_agent,rule_based_agent`
- Er wurde in 3RB und im externen Trio als konkreter Challenger auditiert.
- Der saubere Submission-Ordner ist `agent_code/dqn_task4/`.
- Der aktuelle Incumbent darf nicht überschrieben werden.

### Wichtig: Entwicklungs- versus Submission-Agent

Alle Experimente, Trainingsarme und Entwicklungs-Evaluationen verwenden
`--agents ben_task4` und Dateien unter `agent_code/ben_task4/`. Das ist der
aktive Entwicklungsagent. `agent_code/dqn_task4/` ist ausschließlich der
eingefrorene, minimale Submission-Ordner mit einer Kopie des ausgewählten
Mixed-Kill-2000-Modells; er wird nicht weitertrainiert. Ein `dqn_task4`-Lauf
prüft nur die spätere Abgabeintegration und ist kein Ersatz für `ben_task4`.

Der 7.000-Episoden-Fortsetzungsarm wurde verworfen: Full1000 zeigte Score `−0,370`
und weniger Kills gegenüber dem 2.000er Modell. N-step-3 und mehrere neue Feature-
Varianten wurden ebenfalls beendet.

## Externe Agenten

Externe Agenten dürfen als Trainingsgegner verwendet werden. Sie bleiben reine
Spielgegner und werden nicht in unseren Submission-Agenten kopiert, importiert oder
eingereicht:

- `ext_lijesse_featureeverything`
- `ext_xiaoxiae_bindist_v2`
- `ext_xiaoxiae_binary_v6`

Der frühere External-Trio-Pilot (`external_trio_v1`) startete noch vom älteren
sicheren Modell und ist daher keine faire Kontrolle für den aktuellen Incumbent.
Ein neuer kontrollierter Pilot startete vom Mixed-Kill-2000-Modell:

- Kontrolle: `mixed_external_control_v1`, weiterhin
  `peaceful_agent,rule_based_agent,rule_based_agent`
- Kandidat: `mixed_external_v1`, die drei externen Agenten
- Beide vollständig mit 1.000 Episoden, Seed 11, 11 Kanälen.
- External-Quick100 Kandidat minus Kontrolle: Score `−0,330`
  `[−0,750; +0,100]`, Kills `−0,050` `[−0,120; +0,010]`, Suizide `−0,010`,
  Survival `+0,040`.
- Ergebnis: kein nachgewiesener Vorteil, falsche Richtung bei Score und Kills;
  kein Full1000 und kein weiteres Training dieser Konfiguration.
- Audit: `scratchpad/audit_ben_task4_external_mixed_quick100_20260909.md`.

## Bereits verworfene Task-4-Richtungen

- 7.000er Mixed-Kill-Fortsetzung: schlechter als 2.000er Incumbent.
- N-step-3: Score schlechter, Suizide höher.
- Temporal-Safety-Kanal: keine tragfähige Verbesserung, Sicherheitsproblem.
- Opponent-bomb-tradeoff-Kanal: Score/Kills ohne Effekt, Suizide `+0,180`, Survival
  `−0,160`; Audit:
  `scratchpad/audit_ben_task4_opponent_bomb_tradeoff_quick100_20260908.md`.
- Festes External-Trio-Training vom Mixed-Kill-2000-Modell: Quick100-Gate verfehlt.

## Priorisierte nächste Schritte

Nicht blind weitertrainieren. Zuerst den Incumbent gegner-spezifisch diagnostizieren:

1. Den Mixed-Kill-2000-Agenten einzeln beziehungsweise slotbalanciert gegen
   Li-Jesse, Bindist und Binary messen. Gleiche Arenaseeds, live sichtbarer
   Fortschritt, Score/Coins/Kills/Suizide/killed-by/Survival/think-time.
2. Aus der größten nachgewiesenen Lücke genau eine Hypothese wählen:
   - Coins/Überleben: ein kontrolliertes Navigations-/Sicherheitsfeature.
   - Kill-Conversion: ein einzelnes Angriffssignal mit Escape-Gate.
   - Gegnerverteilung: ein wechselndes Curriculum statt eines festen External-Trios,
     immer mit Nullkontrolle vom selben Mixed-Kill-Checkpoint.
3. Vor jedem Training eine neue Designprüfung und ein adversarialer Audit; danach
   Kontrolle zuerst, Kandidat erst nach Artefaktprüfung.
4. Zusätzlich ist ein Ensemble bestehender Checkpoints möglich, aber nur mit einem
   vorher festgelegten Validierungsset und ohne nachträgliche Bestwertsuche.

Die Diagnose aus Punkt 1 ist abgeschlossen: Gegen Li-Jesse erreicht der Incumbent
Score `3,49`, `0,00` Kills und `0,35` Suizide; gegen Bindist `3,27`, `0,01` Kills
und `0,61` Suizide; gegen Binary `3,17`, `0,02` Kills und `0,65` Suizide.
Die stärkste gemeinsame Lücke ist damit Bombensicherheit/Escape, verbunden mit sehr
geringer Kill-Conversion. Als nächstes wird Punkt 4 (Überleben) vor Punkt 2/3
präzisiert; ein Angriffspilot ohne gelöste Sicherheit wäre derzeit nicht sinnvoll.

Der dafür vorbereitete Pilot heißt `survival_penalty_control_v1` versus
`survival_penalty10_v1`. Beide starten vom Mixed-Kill-2000-Modell und trainieren im
Mixed-Feld; einzig `GOT_KILLED` ist im Kandidaten `-10` statt `-5`. Audit:
`scratchpad/audit_ben_task4_survival_penalty_design_20260909.md`.

Punkt 3 wird nicht als blindes neues Reward-Experiment dupliziert: `safe_offense`
mit Escape-Gate, Gegnerdruck-/Alignment-Features und Killreward-Varianten wurden in
der Historie bereits getestet und entweder nicht nachgewiesen oder wegen
Sicherheitsverlust gestoppt. Ein neuer Angriffspilot ist nur mit einer tatsächlich
neuen lokalen Kill-Gelegenheit plus expliziter Escape-Bedingung zulässig.

Der Ensemble-Test `ensemble_v1` ist als inference-only Validierung vorbereitet. Er
mittelt die Q-Werte des Mixed-Kill-2000-Incumbents und des Curriculum-p4-Modells,
bevor die bestehende Aktionsmaske angewendet wird. Keine Promotion vor 3RB- und
External-Trio-Validierung.

Der Ensemble-Quick100 zeigte keinen nachgewiesenen Score-/Killvorteil: 3RB Score
`3,760`, External-Trio Score `2,450`; die direkte External-Differenz zur verfügbaren
Kontrolle war nur `+0,050 [-0,310, +0,420]`. Kein Full1000 und keine Promotion.

Der finale Ensemble-Full1000 bestätigt den No-Go: gegenüber dem frisch gemessenen
Incumbent Score `-0,289 [-0,489, -0,086]` und Kills `-0,080 [-0,114, -0,046]`,
bei besserer Sicherheit (Suizide `-0,184`, Survival `+0,216`). Wegen des primären
Scoreverlusts bleibt der Mixed-Kill-2000-Agent offizieller Kandidat; das Ensemble
bleibt nur Sicherheits-Fallback.

Der Survival-Penalty-Pilot ist inzwischen negativ abgeschlossen: Kandidat minus
Kontrolle im Quick100 Score `-0,020`, Suizide `+0,100`, Survival `-0,060`; kein
Full1000. Ein stärkerer Todesreward allein löst die Escape-Lücke nicht.

Die Frage nach zu kurzer Trainingszeit ist offen, aber derzeit nicht belegt: Beide
Survival-Arme setzten den 2.000er Incumbent fort, also etwa 3.000 Episoden Gesamt-
erfahrung. In den 1.000er Logs blieb Epsilon bei `0,05`, die späten Loss-Werte waren
stabil (`ca. 0,039` Kontrolle, `ca. 0,052` Kandidat), und die Spätphasen zeigten
keinen konsistenten Scoreaufwärtstrend. Vor einer längeren Fortsetzung zuerst
Checkpoint-Evaluationen bei Episode 100/500/1000; längeres Training nur bei einer
plausiblen Lernkurve und einer vorab definierten neuen Hypothese.

## Bens geplanter Fünf-Punkte-Durchlauf

Die fünf Verbesserungsrichtungen werden sequenziell und mit Gates bearbeitet:

1. Gegner-spezifische Diagnose gegen Li-Jesse, Bindist und Binary.
2. Kontrolliertes wechselndes Curriculum mit externen und Rule-based-Gegnern.
3. Einzelne Kill-Conversion-Hypothese mit Escape-Sicherheitsgate.
4. Einzelne Coins-/Überlebens-Hypothese mit Nullkontrolle.
5. Ensemble bestehender Checkpoints auf einem vorher festgelegten Validierungsset.

Punkt 1 entscheidet, welche der Punkte 2--4 tatsächlich zuerst umgesetzt wird.
Punkt 5 erfolgt zuletzt und nur, wenn die vorhandenen Checkpoints messbar
komplementäre Stärken zeigen. Kein Punkt darf automatisch als Erfolg gelten; jeder
neue Trainingsarm braucht Kontrolle, Audit, sichtbare Evaluation und BEN-Eintrag.

## Relevante Dateien und Befehle

- Tagebuch: `BEN.md`
- Produktionskandidat: `agent_code/dqn_task4/`
- Entwicklungsagent: `agent_code/ben_task4/`
- External-Pilot: `tools/train_ben_task4_external_mixed_pilot.sh`
- Survival-Pilot: `tools/train_ben_task4_survival_penalty_pilot.sh`
- Curriculum-Pilot: `tools/train_ben_task4_curriculum_pilot.sh`
- Ensemble-Validierung: `BM_TASK4_TRAINING_ARM=ensemble_v1` mit `MODEL_VARIANT=baseline`
- Evaluator: `tools/evaluate.py` — immer ohne `--quiet`
- Analyse: `tools/analyze.py`
- Letzte External-Evaluationen: `results/eval/task4_external_mixed/`
- Letzte Tradeoff-Evaluationen: `results/eval/task4_tradeoff/`

Nach jedem neuen Lauf: CSV-Zeilen und Metadaten prüfen, Modellhash sichern,
Provenienz verifizieren, erst dann analysieren und `BEN.md` aktualisieren.

## Plateau-Kontrolle

Training wird künftig nicht anhand von Trainingsscore oder Loss verlängert. Für
jede neue Trainingshypothese werden chronologische greedy-Evaluations-CSV-Dateien
auf denselben Validierungsarenen gesammelt und mit
`uv run python tools/plateau_check.py <csv1> <csv2> ...` geprüft. Standardmäßig
gelten zwei aufeinanderfolgende Scorezuwächse unter `0,10` als Plateau; eine
Suizidsteigerung über `0,03` bleibt unabhängig davon ein Sicherheitsfehler. Zu
wenige Phasen führen zu `PLATEAU=insufficient_history`.

Der phasenweise Kill-Conversion-Pilot ist als
`tools/train_ben_task4_kill_reward_plateau_pilot.sh` vorbereitet. `control` und
`candidate` verwenden jeweils vier 250-Episoden-Phasen vom selben Mixed-Kill-
2000-Checkpoint, alternieren Mixed- und External-Trio-Gegner und evaluieren nach
jeder Phase greedy im External-Trio. Der Kandidat ändert ausschließlich den
internen Trainingsreward für `KILLED_OPPONENT` von `+5` auf `+7,5`; der offizielle
Evaluationsscore bleibt `+5`. Design-Audit:
`scratchpad/audit_ben_task4_kill_reward_plateau_design_20260910.md`.

Der Lauf ist abgeschlossen. Im letzten Quick100 liegt der Kandidat bei Score
`2,280` gegenüber `1,820` für die Kontrolle; die gepaarte Differenz ist
`+0,460 [+0,150,+0,780]`. Kills sind mit `+0,030 [0,+0,070]` nicht nachgewiesen,
und die Phasenkurven sind nicht monoton. Beide Arme melden
`PLATEAU=still_improving`, was hier nur noisy Phasenwechsel bedeutet.
Audit:
`scratchpad/audit_ben_task4_kill_reward_plateau_results_20260910.md`.

Der Kandidat ist damit ein vorläufiger Full1000-Challenger, aber noch kein
offizieller Ersatz. Eine Promotion setzt eine nicht-fragile Scoreverbesserung,
keinen Killverlust und keine Safety-Regression voraus.

Der Full1000-Test ist abgeschlossen und verwirft die Promotion: Kandidat minus
Kontrolle erreicht Score `+0,103 [-0,018,+0,226]`, Kills `-0,003
[-0,020,+0,014]`, Coins `+0,118 [+0,039,+0,198]` und Survival `+0,050
[+0,008,+0,093]`. Audit:
`scratchpad/audit_ben_task4_kill_reward_plateau_full1000_20260911.md`.
Der Mixed-Kill-2000-Incumbent bleibt aktiv; der Kill-Reward-Pilot ist beendet.

## Nächster Einzelpilot: event-balanciertes Replay

Der neue Pilot verändert ausschließlich die Stichprobe aus dem Replay Buffer:
`event_replay_control_v1` bleibt uniform, `event_replay_balanced_v1` sampelt
seltene Kill-Übergänge (ca. 20 %) und Todesübergänge (ca. 15 %) gezielt häufiger.
Event-Tags werden über N-step-Returns und Augmentation erhalten. Rewards,
Features, Modell und Inferenz bleiben gleich. Audit:
`scratchpad/audit_ben_task4_event_replay_design_20260911.md`.

Manueller Start:

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_event_replay_pilot.sh control
```

Danach separat:

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_event_replay_pilot.sh candidate
```

Evaluation erfolgt anschließend automatisch sichtbar über 100 External-Trio-
Runden. Kein Full1000 ohne positives Quick100-Gate.

Der Pilot ist negativ abgeschlossen: Balanced minus uniform im Quick100 ergibt
Score `−0,100 [−0,640,+0,440]`, Coins `−0,450 [−0,740,−0,150]`, Kills
`+0,070 [−0,010,+0,150]` und Suizide `+0,110 [−0,030,+0,250]`. Kein Full1000,
keine Promotion. Audit:
`scratchpad/audit_ben_task4_event_replay_quick100_20260911.md`.

## Nächster Einzelpilot: niedrigere Lernrate

`learning_rate_control_v1` nutzt die bisherige Lernrate `1e-4`, während
`learning_rate5e5_v1` nur `5e-5` verwendet. Sonst sind Quelle, Seed, Gegnerfeld,
Rewards, Features, Replay, Target-Update und Exploration gleich. Design-Audit:
`scratchpad/audit_ben_task4_learning_rate_design_20260911.md`.

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_learning_rate_pilot.sh control
```

Danach in einem separaten Terminal:

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_learning_rate_pilot.sh candidate
```

Nach Abschluss eines Arms prüft die Dauerdiagnose die gespeicherten Checkpoints
200/500/800/1000 auf denselben External-Trio-Arenen:

```bash
zsh tools/evaluate_ben_task4_duration_checkpoints.sh learning_rate_control_v1
```

beziehungsweise `learning_rate5e5_v1`. Ein später stabiler Scoreanstieg ohne
Suizidregression rechtfertigt erst dann einen neuen längeren Kontrollversuch;
eine flache/fallende Kurve beendet unveränderte Verlängerungen. Audit:
`scratchpad/audit_ben_task4_duration_checkpoint_design_20260911.md`.

Der niedrige-Lernraten-Kandidat ist im Quick100 verworfen: Score `−0,320`,
Kills `−0,010`, Suizide `+0,060` gegenüber `1e-4`; kein Full1000. Audit:
`scratchpad/audit_ben_task4_learning_rate_quick100_20260911.md`.

Die Dauerdiagnose ist abgeschlossen: Beim `1e-4`-Kontrollarm steigt der Score
von `1,910` (Episode 200) auf `2,490` (500), bleibt danach aber bei `2,460`
(800) und `2,510` (1.000); `PLATEAU=plateau`. Der `5e-5`-Arm fällt spät auf
`2,190` und hat zusätzlich eine Safety-Regression. Keine unveränderte längere
Fortsetzung starten. Audit:
`scratchpad/audit_ben_task4_duration_checkpoint_results_20260911.md`.

## Nächster Einzelpilot: Double DQN

`double_dqn_control_v1` behält das klassische DQN-Target; `double_dqn_v1`
wählt die nächste legale Aktion mit dem Online-Netz und bewertet sie durch das
Target-Netz. Alle anderen Bedingungen sind identisch. Audit:
`scratchpad/audit_ben_task4_double_dqn_design_20260911.md`.

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_double_dqn_pilot.sh control
```

Danach separat:

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_double_dqn_pilot.sh candidate
```

Der Double-DQN-Pilot ist im Quick100 beendet: Score `+0,120` und Suizide
`−0,100` sind nicht nachgewiesen, Kills liegen bei `−0,010`. Kein Full1000,
keine Promotion. Audit:
`scratchpad/audit_ben_task4_double_dqn_quick100_20260911.md`.

## Nächster Einzelpilot: Prioritized Experience Replay

`per_v1` priorisiert Replay-Übergänge über den absoluten TD-Fehler, korrigiert
den Sampling-Bias mit Importance Weights und aktualisiert Prioritäten nach jedem
Update. `per_control_v1` bleibt uniform. Audit:
`scratchpad/audit_ben_task4_per_design_20260911.md`.

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_per_pilot.sh control
```

Danach separat:

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_per_pilot.sh candidate
```

PER ist im Quick100 ohne Effekt beendet: Score `−0,050`, Kills `−0,010`, keine
nachgewiesene Verbesserung. Kein Full1000. Audit:
`scratchpad/audit_ben_task4_per_quick100_20260911.md`.

## Nächster Architekturpilot: Dueling DQN

Der Dueling-Kandidat übernimmt CNN und Hidden-Layer Q-erhaltend vom Incumbent;
sein Value-Head wird aus dem Mittel des alten Q-Heads und der Advantage-Head aus
dem alten Q-Head initialisiert. Daher beginnt er mit exakt derselben Policy.
Kontrolle und Kandidat trainieren danach je 2.000 Episoden im selben Mixed-Feld.
Audit: `scratchpad/audit_ben_task4_dueling_design_20260911.md`.

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_dueling_pilot.sh control
```

Danach separat:

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_dueling_pilot.sh candidate
```

Der Dueling-Pilot ist im Quick100 beendet: Score `+0,090` nicht nachgewiesen,
Kills `−0,010`, Suizide `+0,110`. Kein Full1000 und keine Promotion. Audit:
`scratchpad/audit_ben_task4_dueling_quick100_20260912.md`.

## Letzter neuer Architekturpilot: Auxiliary Opponent Prediction

Der Kandidat behält den Q-Kopf des Incumbents exakt und lernt zusätzlich während
des Trainings die Gegnerbelegung im Folgezustand. Der Hilfskopf beeinflusst nur
die Repräsentation, nie die Inferenzaktion. Audit:
`scratchpad/audit_ben_task4_opponent_prediction_design_20260912.md`.

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_opponent_prediction_pilot.sh control
```

Danach separat:

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_opponent_prediction_pilot.sh candidate
```

Auxiliary Opponent Prediction ist im Quick100 verworfen: Score `−0,100`, Kills
`+0,020` nicht nachgewiesen, Suizide `+0,050`. Kein Full1000. Audit:
`scratchpad/audit_ben_task4_opponent_prediction_quick100_20260912.md`.

## Nächster Schritt: Mixed-Kill-Mehrseedreplikation

Keine neue Mechanik: Seeds 12 und 13 reproduzieren das ursprüngliche
Mixed-Kill-2000-Protokoll. Beide können parallel starten und evaluieren danach
sichtbar in 3RB sowie External-Trio. Audit:
`scratchpad/audit_ben_task4_mixed_kill_multiseed_design_20260912.md`.

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_mixed_kill_multiseed.sh 12
```

Im zweiten Terminal:

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_mixed_kill_multiseed.sh 13
```

Beide Replikate sind abgeschlossen und im Quick100 verworfen. Seed 13 liegt in
3RB nominell bei `+0,510 [−0,060,+1,080]` Score gegenüber Seed 12, aber ohne
Nachweis; im External-Trio sind es nur `+0,050 [−0,380,+0,510]` und zugleich
nachweisbar mehr Suizide (`+0,270 [+0,140,+0,400]`). Das ist Trainingsvarianz,
kein bester Seed: kein Full1000, keine Promotion. Audit:
`scratchpad/audit_ben_task4_mixed_kill_multiseed_quick100_20260913.md`.

Ben startet manuell, beispielsweise:

```bash
cd /Users/bengschwend/Projects/bomberman_RL
zsh tools/train_ben_task4_kill_reward_plateau_pilot.sh control
```

Der Kandidat kann in einem zweiten Terminal mit `candidate` gestartet werden,
weil die Artefaktpfade getrennt sind. Nicht denselben Arm parallel starten.

## Aktueller Abschlussplan

Keine weiteren Trainingsvarianten starten. Den Submission-Ordner `agent_code/dqn_task4`
mit dem Incumbent-SHA `d50ae3...06c6` einfrieren, einen lokalen CPU-1-Runden-Smoke-Test
ausführen, danach optional eine letzte Submission-Agent-Evaluation in 3RB und im
External-Trio durchführen. Anschließend nur noch Bericht, Experimentledger und
Abgabearchiv vorbereiten; kein Ensemble und kein Curriculum-Modell in den offiziellen
Ordner übernehmen.
