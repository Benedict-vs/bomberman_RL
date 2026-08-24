# Bens Logbuch

Persönliches Arbeitstagebuch: Wo ich stehengeblieben bin, woran ich gerade arbeite,
welche Ideen ich habe und warum ich mich so entschieden habe.

---

## Einträge

<!-- Neueste Einträge oben. Format:
### JJJJ-MM-TT — Kurztitel
- **Stand:** wo ich aufgehört habe
- **Gemacht:** was passiert ist
- **Entscheidung + Warum:** ...
- **Nächster Schritt / Ideen:** ...
-->

### 2026-08-24 — Langen Task-2-Lauf bei Episode 7.000 beendet
- **Gemacht:** Der ursprünglich auf 15.000 Episoden ausgelegte `safety_potential_v1`-Lauf wurde unmittelbar nach der vollständig protokollierten Episode 7.000 kontrolliert beendet; Endmodell, CSV, final aktualisiertes Live-Diagramm und Episode‑7000‑Checkpoint sind vorhanden. Der Prozess erhielt erst nach dem Schreiben des Checkpoints das geplante Interrupt-Signal.
- **Warum:** ε hatte bereits `0,05` erreicht und die verbleibenden 8.000 Episoden hätten wegen stark verlängerter Trajektorien mehrere zusätzliche Stunden beansprucht. Episode 7.000 liefert genügend Training bei minimaler Exploration für eine erste greedy Evaluation, bevor weitere Rechenzeit investiert wird.

### 2026-08-24 — Live-Diagramm in den Task-2-Agenten verschoben
- **Gemacht:** Das laufend aktualisierte Trainingsdiagramm liegt nun unter `agent_code/ben_task2/ben_task2_safety_potential_v1_15000ep_seed11_live.png`; die zuvor unter `results/train/ben_task2/` erzeugte Datei wurde dorthin verschoben und der Plotter auf das neue Ziel umgestellt. Die eigentlichen CSV-, Metadaten- und Checkpointdateien bleiben unverändert in `results/train/ben_task2/`.
- **Warum:** Das Diagramm soll nach Wunsch direkt beim Task‑2-Agenten auffindbar sein, während die umfangreichen Trainingsartefakte weiterhin getrennt im Ergebnisordner liegen. Der eindeutige Dateiname verhindert Kollisionen mit bestehenden Modellen und früheren Diagrammen.

### 2026-08-24 — Live-Diagramm für Trainingsfortschritt
- **Gemacht:** Das neue Werkzeug `tools/live_training_plot.py` beobachtet ein Trainings-CSV und aktualisiert nach jedem Block von 100 Episoden atomar ein Diagramm mit Münzen/Kisten, Suiziden/Überleben, Schritten, Bomben/WAIT, ε und Loss. Für den laufenden Task‑2-Versuch erhält das Diagramm einen eigenen Speicherort unter `results/train/ben_task2/`.
- **Warum:** Der lange Lauf soll während des Trainings anhand geglätteter Verhaltenstrends beurteilt werden können, statt erst nach mehreren Stunden oder anhand stark schwankender Einzelrunden. Die 100-Episoden-Blöcke passen zur Checkpoint-Frequenz und machen Fortschritt oder Policy-Kollaps früh sichtbar.

### 2026-08-24 — Langer Flucht-Potential-Lauf vorbereitet
- **Gemacht:** Die unveränderte `safety_potential_v1`-Konfiguration erhält für den geplanten Lauf über 15.000 Episoden eigene Modell-, Checkpoint- und Trainingslognamen mit `15000ep_seed11`. Die Konfigurationstests wurden auf diese neuen Task‑2-Speicherziele angepasst.
- **Warum:** Nach 1.000 Episoden lagen nur 9.454 Übergänge vor, davon lediglich 4.454 nach dem Replay-Warm-up, und ε war noch ungefähr `0,91`; der negative greedy Befund kann daher schlicht zu frühes Training widerspiegeln. Der längere Lauf soll die bestehende Reward-Zuordnung fair testen, ohne frühzeitig einen potenziell schädlichen direkten Bombenreward einzuführen.

### 2026-08-24 — Flucht-Potential greedy evaluiert
- **Gemacht:** Das nach 1.000 Episoden trainierte `safety_potential_v1` wurde greedy auf 100 festen Task‑2-Arenen ausgewertet. Der Agent hatte `0` ungültige Aktionen und `0` Suizide, legte aber auch `0` Bomben, zerstörte `0` Kisten, sammelte `0` Münzen und erreichte in allen 100 Runden das 400-Schritte-Timeout bei Score `0`.
- **Warum:** Die Evaluation sollte prüfen, ob das dichte Sicherheitssignal außerhalb der explorativen Trainingspolicy nutzbares Flucht- und Bombenverhalten erzeugt. Stattdessen verstärkt es lediglich eine sichere Bewegungs-/Wartepolicy; als nächstes muss der fehlende Anreiz zum kontrollierten Bombenlegen beziehungsweise dessen zeitliche Reward-Zuordnung untersucht werden.

### 2026-08-24 — Flucht-Potential über 1.000 Episoden trainiert
- **Gemacht:** Die Variante `safety_potential_v1` wurde mit Seed `11` über 1.000 Task‑2-Runden ohne Gegner trainiert; alle Modelle, Checkpoints, Logs und der Trainingsplot liegen unter eigenen, kollisionsfreien Task‑2-Namen. In den letzten 100 Episoden zerstörte sie im Mittel `3,42` Kisten und hatte `0` ungültige Aktionen, endete aber weiterhin zu `100 %` durch Suizid und überlebte keine Runde.
- **Warum:** Der Lauf sollte prüfen, ob das zustandsbasierte Sicherheitssignal nach dem Replay-Warm-up das Fliehen vor eigenen Bomben lernbar macht. Die Trainingsdaten zeigen dafür noch keinen Fortschritt; als nächstes muss eine greedy Evaluation klären, welches deterministische Verhalten das Netz tatsächlich gelernt hat.

### 2026-08-24 — Eigene Speicherziele für den Flucht-Potential-Lauf
- **Gemacht:** Modell, Checkpoints und Trainingsprotokoll der neuen Variante verwenden nun den eindeutigen Namen `safety_potential_v1_1000ep_seed11`; vorhandene Legal-Mask- und Reward-Artefakte bleiben unverändert. Die zugehörigen Konfigurationstests wurden an die neuen Task-2-Namen angepasst.
- **Warum:** Der bisherige Zielname zeigte noch auf den bereits abgeschlossenen Legal-Mask-Lauf und hätte dessen Daten beim nächsten Training überschreiben können. Ein eigener Variantenname hält den Vergleich reproduzierbar und schützt alle bisherigen Resultate.

### 2026-08-24 — Zustandsbasiertes Flucht-Potential ergänzt
- **Gemacht:** Zusätzlich zum deaktivierten Coin-Potential verwendet der Task‑2-DQN nun ein Sicherheits-Potential `Φ(s) = 1 − Gefahr_am_eigenen_Feld` mit Skala `1,0`; Flucht wird positiv, zunehmende Gefahr und ein terminaler Tod werden negativ geformt. Tests decken das Verlassen und Betreten einer Bombenlinie sowie den terminalen Fall ab.
- **Warum:** Die Maskenvariante lernte zwar gültige Bewegungen, aber weder gezieltes Bombenlegen noch zuverlässiges Fliehen. Das zustandsbasierte Potential liefert ein dichtes Lebensrettungssignal, ohne eine konkrete Aktion oder Fluchtrichtung regelbasiert vorzugeben.

### 2026-08-24 — Legal-Action-Maske trainiert und greedy evaluiert
- **Gemacht:** Die Maskenvariante wurde mit Seed `11` über 1.000 Episoden trainiert und auf denselben 100 Task‑2-Arenen wie die unmaskierte Basislinie greedy evaluiert. Invalid Actions fielen im Training auf exakt `0`; greedy führte der Agent in jeder Runde 400 gültige Bewegungen aus, legte aber keine Bombe, zerstörte keine Kiste und sammelte keine Münze.
- **Warum:** Der kontrollierte Vergleich sollte prüfen, ob weniger ungültige Erfahrung die WAIT-Policy aufbricht und nutzbares Verhalten erzeugt. Die Maske beseitigte WAIT und Suizide, ersetzte sie jedoch nur durch eine Bewegungsschleife bis zum Timeout; `100 %` Überleben ist daher kein Task‑2-Fortschritt und Score blieb unverändert bei `0`.

### 2026-08-24 — Legal-Action-Maske für den Task-2-DQN
- **Gemacht:** Exploration, greedy Auswahl und Bellman-Target berücksichtigen nun dieselbe Legal-Action-Maske: Bewegungen in Steinwände, Kisten oder Bomben sowie `BOMB` ohne verfügbare Bombe werden ausgeschlossen, `WAIT` bleibt erlaubt. Neue Tests prüfen die Maskengeometrie, die Aktionsauswahl und das Ignorieren eines illegalen maximalen Q-Werts im Target.
- **Warum:** Im ersten 1.000-Episoden-Lauf waren rund die Hälfte der Aktionen ungültig und die greedy Policy kollabierte vollständig auf `WAIT`. Die Maske soll den Replay Buffer mit mehr gültigen Bewegungsübergängen füllen und verhindern, dass hohe Q-Werte unmöglicher Aktionen Lernen und Inferenz verzerren.

### 2026-08-24 — Greedy Evaluation der Task-2-DQN-Basislinie
- **Gemacht:** Das nach 1.000 Episoden trainierte Modell wurde greedy über 100 `classic`-Runden ohne Gegner mit dem festen Evaluationsseed gemessen. Die Policy erzielte `0` Score und Münzen sowie exakt `0` Bewegungen; in 72 Runden wartete sie bis zum 400-Schritte-Limit, in 28 Runden legte sie je eine Bombe und starb daran.
- **Warum:** Die Evaluation sollte klären, ob der sinkende Trainings-Loss in nutzbares Verhalten übersetzt wurde. Die nominelle Überlebensrate von `72 %` ist nur eine WAIT-Schleife und kein Fortschritt; jede gelegte Bombe führte zum Suizid, sodass die Task‑2-Basislinie in dieser Form verworfen beziehungsweise grundlegend diagnostiziert werden muss.

### 2026-08-24 — Task-2-Trainingsartefakte zentral abgelegt
- **Gemacht:** Trainings-CSVs, Metadaten, Plot, Pilotmodell und sämtliche Checkpoints wurden nach `results/train/ben_task2/` verschoben; zukünftige Trainingslogs und Checkpoints werden direkt dort gespeichert. Nur das aktuell für Inferenz und Einreichung benötigte Endmodell bleibt in `agent_code/ben_task2/`.
- **Warum:** Die Trennung hält den eingereichten Agenten frei von historischen Trainingsdaten und verhindert, dass Task‑2-Artefakte zwischen allgemeinen Ergebnisdateien oder Agentcode verstreut liegen. Der eigene Unterordner macht außerdem Kollisionen mit Task 1 und anderen Agenten leichter erkennbar.

### 2026-08-24 — Task-2-Basislinie über 1.000 Episoden trainiert
- **Gemacht:** Die Task‑2-DQN-Basislinie wurde mit Seed `11` für 1.000 `classic`-Runden ohne Gegner von Grund auf trainiert und unter neuen Modell-, Checkpoint-, Log- und Metadatennamen gespeichert. Ab Episode `457` fanden Netzwerkupdates statt; insgesamt enthalten 544 Episoden einen Loss, der im letzten 100er-Fenster im Mittel bei `0,097` lag.
- **Warum:** Der längere Lauf sollte das Replay-Warm-up sicher überschreiten und zeigen, ob die neuen Gefahrenkanäle und Basis-Rewards erste Flucht- oder Münzsammelmuster erzeugen. Dieses frühe Lernsignal blieb aus: Auch die letzten 100 Episoden endeten zu 100 % als Suizid und ohne Münze, sodass die Trainingskurve trotz sinkendem Loss noch keinen Verhaltensfortschritt zeigt; eine greedy Evaluation steht aus.

### 2026-08-24 — Erster Task-2-DQN-Pilotlauf
- **Gemacht:** Der Task‑2-DQN wurde mit Seed `11` für 300 `classic`-Runden ohne Gegner trainiert; Modell, drei Checkpoints und ein eigenes Trainingsprotokoll wurden unter kollisionsfreien Task‑2-Namen gespeichert. Der zufällige Agent zerstörte im Mittel `2,90` Kisten, starb aber in allen 300 Runden durch die eigene Bombe und sammelte keine Münze.
- **Warum:** Der Pilot sollte vor einem längeren Lauf prüfen, ob Zustände, Rewards, terminale Todesübergänge und Artefaktspeicherung technisch funktionieren. Mit nur `3.282` Replay-Übergängen blieb der Buffer unter dem Warm-up von `5.000`, sodass noch kein Netzwerkupdate stattfand und der Lauf keine Aussage über Lernleistung erlaubt.

### 2026-08-24 — Coin-Potential für die Task-2-Basislinie deaktiviert
- **Gemacht:** `POTENTIAL_REWARD_SCALE` wurde für den ersten Task‑2-Basislauf von `1,0` auf `0,0` gesetzt und die Tests sichern nun ab, dass Annäherung und Entfernung zu sichtbaren Münzen keinen zusätzlichen Shaping-Reward erzeugen. Die BFS-basierte Potentialberechnung bleibt im Code und separat getestet.
- **Warum:** Das aus Task 1 übernommene Potential berücksichtigt nur sichtbare Münzen und liefert für unter Kisten verborgene Münzen kein geeignetes Signal. Ohne dieses Shaping lässt sich zunächst kontrolliert messen, was die neuen Task‑2-Features und Basis-Rewards allein lernen.

### 2026-08-24 — Terminale Todes-Rewards mit Integrationstests abgesichert
- **Gemacht:** Zwei Unit-Tests simulieren nun den Tod durch eine fremde Bombe und einen Suizid über `end_of_round` und prüfen den vollständigen terminalen Replay-Übergang. Beide Fälle müssen genau `−5,05` Reward erhalten, als terminal markiert sein und dürfen keinen Folgezustand besitzen.
- **Warum:** Die reine Rewardtabelle konnte nicht nachweisen, dass Todes-Events tatsächlich im Replay Buffer ankommen. Die neuen Tests sichern insbesondere ab, dass `GOT_KILLED` und `KILLED_SELF` bei einem Suizid nicht zu einer doppelten Todesstrafe führen.

### 2026-08-24 — Terminalen Todesübergang im Replay Buffer speichern
- **Gemacht:** `end_of_round` speichert bei `GOT_KILLED` nun die zuvor fehlende letzte Aktion als terminalen Replay-Übergang und übernimmt dabei alle terminalen Events samt Reward. Bei einem überlebten Rundenende wird weiterhin nur der bereits gespeicherte letzte Übergang als terminal markiert.
- **Warum:** Das Framework ruft `game_events_occurred` nach dem Tod nicht mehr auf, weshalb die tödliche Aktion und damit auch der neue Todesmalus bisher nie im Replay Buffer ankamen. Ein eigener terminaler Übergang ordnet die Strafe der tatsächlich zum Tod führenden Aktion zu, statt fälschlich die vorherige Aktion zu bestrafen.

### 2026-08-24 — Kürzerer Trainingsseed für Task 2
- **Gemacht:** Der feste Trainingsseed des Task‑2-DQN wurde von `20260805` auf `11` geändert und der Run-Name entsprechend auf `task2_reward_v1_seed11` verkürzt. Ein Unit-Test sichert sowohl den Seedwert als auch seine Kennzeichnung im Run-Namen ab.
- **Warum:** Der kurze Seed hält zukünftige Namen von Trainingslogs und Checkpoints kompakter, ohne auf reproduzierbare Initialisierung zu verzichten. Die Änderung gilt ausschließlich für den Task‑2-Agenten und berührt keine bisherigen Task‑1-Artefakte.

### 2026-08-24 — Task-2-Agent vollständig vom Task-1-DQN getrennt
- **Gemacht:** Alle Tests in `agent_code/ben_task2` importieren nun ausdrücklich die Task‑2-Module; die verbliebenen direkten und unqualifizierten Task‑1-Importe wurden entfernt. Auch die Modulbeschreibungen von Training und Inferenz bezeichnen den Agenten jetzt eindeutig als Task‑2-DQN.
- **Warum:** Einige Tests prüften unbemerkt noch Implementierungen aus `ben_coin_collector_dqn`, wodurch Fehler im neuen Agenten unentdeckt bleiben und alte Daten oder Module versehentlich verwendet werden konnten. Explizite Task‑2-Imports stellen sicher, dass Entwicklung und Tests vollständig voneinander getrennt sind.

### 2026-08-24 — Eigene Speicherziele für den Task-2-DQN
- **Gemacht:** Modell, Checkpoints, Trainings-CSV und Metadaten des Task‑2-DQN verwenden nun eindeutig benannte Task‑2-Speicherziele statt der aus Task 1 übernommenen Namen. Ein Unit-Test stellt sicher, dass Modell- und Run-Name weiterhin `task2` enthalten.
- **Warum:** Die alten Namen konnten vorhandene Task‑1-Modelle und Trainingsprotokolle überschreiben, wie es beim Testlauf bereits mit einer Metadatendatei passiert ist. Getrennte Namen schützen die bisherigen Versuchsergebnisse und machen neue Artefakte eindeutig zuordenbar.

### 2026-08-24 — Basis-Rewards für Task 2
- **Gemacht:** Das DQN belohnt nun zerstörte Kisten mit `+0,2` und bestraft jeden Tod einmalig mit `−5`; die bestehenden Münz-, Schritt- und Invalid-Action-Rewards bleiben unverändert. Die neuen Rewards und insbesondere die gemeinsame Auslösung von `GOT_KILLED` und `KILLED_SELF` bei einem Suizid werden durch Unit-Tests abgesichert.
- **Warum:** Der Agent benötigt ein Lernsignal, um Kisten gezielt mit Bomben zu öffnen und gleichzeitig das Überleben zu priorisieren. Der vollständige Todesmalus liegt auf `GOT_KILLED`, während `KILLED_SELF` bei `0` bleibt, damit ein Suizid nicht doppelt bestraft wird.

### 2026-08-21 — Bombenaktion im Task-2-Aktionsraum
- **Gemacht:** `BOMB` wurde als sechste Aktion in das DQN aufgenommen und der Ausgang des Netzes entsprechend erweitert. Die Symmetrie-Augmentierung lässt die richtungsneutralen Aktionen `BOMB` und `WAIT` unverändert; Modell-, Callback- und Augmentierungstests wurden angepasst.
- **Warum:** Der Agent muss in Task 2 selbst Bomben legen können, um Kisten zu zerstören und darin verborgene Münzen freizulegen. Die korrekte Behandlung bei der Augmentierung verhindert, dass aus einer Bombenaktion durch Drehen oder Spiegeln fälschlich eine Bewegungsaktion wird.

### 2026-08-21 — Task-2-Zustandsdarstellung
- **Gemacht:** Die CNN-Eingabe von drei auf acht Kanäle erweitert: Wände, Kisten, Münzen, eigene Position, normalisierte Bombentimer, aktive Explosionen, berechnete Bombengefahr und Bombenverfügbarkeit. Das Modell und die zugehörigen Feature- und Modelltests wurden an die neue Eingabe angepasst.
- **Warum:** Der Task-1-Zustand enthält nicht die Informationen, die der Agent zum Öffnen von Kisten und zur Flucht vor Bomben benötigt. Bombentimer und Gefahr werden getrennt dargestellt, damit das Netz sowohl die Bomben selbst als auch deren zeitlich und räumlich wirksame Explosionsbereiche erkennen kann.
