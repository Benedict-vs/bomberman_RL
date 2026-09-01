# Adversarial audit: Ben Task 3 `peaceful_v1`, 2026-08-29

## Auftrag und vorläufiges Urteil

Geprüft wird die Behauptung, dass `ben_task3_peaceful_v1_5000ep_seed11.pt` gegenüber der 11-Kanal-Task-2-Baseline ein klar stärkerer praktischer Task-3-Kandidat gegen drei `peaceful_agent` ist. Ich habe gezielt nach einer Widerlegung gesucht.

**Vorläufiges Urteil:** Die konkrete trainierte Policy ist in den vorliegenden 1000-Runden-Läufen klar stärker. Dies lässt sich nicht auf einen einfachen Auswertungs-, Modellzuordnungs- oder Zählerfehler zurückführen. Der breitere Claim ist jedoch zu eng zu formulieren: Die Daten belegen weder eine robuste Verbesserung über Trainingsseeds noch die kausale Wirkung des Gegnerkanals allein, keinen globalen Task-3-Beststand und keine Freeze-/Submission-Reife.

## Geprüfte Artefakte

- Baseline-Evaluation: `results/eval/task3_opponents/ben_dqn_task3_11ch_task2_baseline_seed11__task3_peaceful_eval1000.csv` plus Meta-Datei
- Kandidaten-Evaluation: `results/eval/task3_opponents/ben_dqn_task3_peaceful_v1_5000ep_seed11__task3_peaceful_eval1000.csv` plus Meta-Datei
- Baseline-Modelle mit 10 und 11 Kanälen, trainiertes Endmodell und Episode-5000-Checkpoint
- Trainings-CSV und Trainings-Metadaten
- `callbacks.py`, `features.py`, `model.py`, `train.py` sowie die Auswertung mit `tools/analyze.py --compare ... --preset task3 --markdown`

## Integrität der Evaluationen

Beide CSVs enthalten exakt 4000 Zeilen: je 1000 Zeilen für `ben_task3` und je 1000 für jeden der drei Gegner-Slots. Die Schlüssel `(round, seed, slot)` sind innerhalb jedes Laufs vollständig und eindeutig. Zeilenreihenfolge, Runde, Seed, Slot und Agentenname stimmen zwischen den beiden Dateien vollständig überein; die Seeds laufen lückenlos von 20260731 bis 20261730. Das ist eine korrekte Paarung der **Arenen und Slots**.

Die Gegneraktionen sind dagegen nicht reproduziert. Nur 2 von 1000 Runden hatten über alle drei Gegner hinweg dieselben spielbezogenen Gegnerresultate. Das entspricht dem bekannten Problem, dass `peaceful_agent` auch den nicht vom Evaluator gesetzten stdlib-Zufall verwendet. Der Vergleich ist deshalb nicht „dieselben Spiele mit anderer Policy“, sondern „dieselben Arenen mit neuen Gegner-Trajektorien“. Eine gepaarte Differenz pro Arena bleibt eine sinnvolle Schätzung für genau diese beiden unabhängigen Läufe und ihre CI enthält die dabei beobachtete Gegnerstreuung; die Paarung darf aber nicht als vollständige deterministische Kontrolle ausgegeben werden. Eine einzelne Wiederholung kann den Gegner-Zufall nicht als Replikationsrisiko ausschließen.

Die Metadaten stimmen in Szenario, 1000 Runden, Basisseed, Agentenaufstellung und Framework-Regeln überein. Beide tragen denselben Callback-SHA `9e0a1634...a5390` und `git_commit: 39c770a-dirty`. Die Dirty-Markierung verhindert eine vollständig commit-reproduzierbare Provenienz: Der Callback-Hash fixiert zwar die wichtigste Inferenzdatei, aber nicht sämtliche uncommitteten Abhängigkeiten wie `features.py` und `model.py`. Die zwei Läufe stammen unmittelbar aus derselben Dirty-Arbeitskopie und sind intern vergleichbar; langfristige Reproduktion nur anhand des Commit-Hashes ist nicht garantiert.

## Metriken und statistischer Claim

Für sämtliche 8000 Agentenzeilen gilt exakt `score == coins + 5*kills`; es gibt keine Rohscore-Abweichung. Beim Ben-Agenten stimmen die Killzähler exakt mit den gegnerischen `killed_by_opponent`-Zählern überein: 1703 in der Baseline und 2651 beim Kandidaten. Da `peaceful_agent` keine Bomben legt, sind seine Todesfälle vollständig den Ben-Kills zugeordnet. Beim Ben-Agenten gilt in allen Runden `survived + died == 1`; Suizide und Todesfälle betragen 135/135 beziehungsweise 81/81. Es gibt damit keinen Hinweis auf einen Zähler- oder Death-Semantikfehler in diesen Runs.

Die offizielle Analyse ergibt für Kandidat minus Baseline:

- Score: +3.366, 95-%-CI [+2.829, +3.899], sign-flip p < 0.0001
- Kills: +0.948, CI [+0.868, +1.026], p < 0.0001
- Suizide: -0.054, CI [-0.081, -0.027], p = 0.0003
- Überleben: +0.054, CI [+0.027, +0.081], p = 0.0003
- ungültige Aktionen: -0.077, CI [-0.119, -0.035], p = 0.0003

Keine dieser Task-3-Zeilen ist als `(fragile)` markiert. Der Scoregewinn wird hauptsächlich durch fast einen zusätzlichen Kill pro Runde getragen und überkompensiert die bereits berichteten Rückgänge bei Münzen und Kisten. Auf **diesen beiden Läufen** ist der praktische Policygewinn groß und statistisch eindeutig. Wegen der nur teilweise gepaarten Gegnertrajektorien sollte die CI nicht als isolierter Trainingseffekt mit vollständig kontrollierter Umwelt beschrieben werden.

## Modell- und Trainingsprovenienz

Die Baseline-Meta-Datei weist ausdrücklich `model_variant: baseline`, Modell `ben_task3_task2_baseline_11ch_seed11.pt` und SHA-256 `b918737e...11399` aus. Die Kandidaten-Meta-Datei weist `model_variant: trained`, Modell `ben_task3_peaceful_v1_5000ep_seed11.pt` und SHA-256 `cbf1f05a...d2e95f` aus. Diese Hashes entsprechen den tatsächlich vorhandenen Dateien. `callbacks.py` erzwingt bei den Varianten `baseline` und `trained` genau diese Auswahl; ein fehlendes trainiertes Modell führt bei `trained` zu einem Fehler. Es gibt keinen Hinweis, dass in den Evaluationen versehentlich dasselbe oder das falsche Modell geladen wurde.

Die Konversion von der 10-Kanal-Task-2-Datei (SHA `b3f27565...671e0`) zur 11-Kanal-Baseline ist verlustfrei: Bei der einzigen geänderten Tensorform, dem ersten Convolution-Gewicht, sind alle zehn alten Kanalscheiben bitidentisch und die neue elfte Scheibe enthält ausschließlich Nullen. Alle übrigen gemeinsamen Tensoren sind bitidentisch. Damit verhält sich die 11-Kanal-Baseline anfänglich unabhängig vom Gegnerkanal wie das Task-2-Netz, soweit Feature- und Maskierungslogik dies zulassen.

Die Trainings-CSV enthält genau 5000 eindeutige Episoden von 1 bis 5000. Das Endmodell und der Episode-5000-Checkpoint haben auf State-Dict-Ebene identische Schlüssel und bitidentische Tensoren. Die Trainings-Meta-Datei nennt die richtige Quelle `ben_task3_task2_baseline_11ch_seed11.pt`, 11 Eingangskanäle, Seed 11, 5000 geplante Episoden und das richtige Zielmodell. Der Lauf ist somit vollständig und das evaluierte Endmodell entspricht dem finalen Checkpoint.

Eine Provenienzschwäche bleibt: Die Trainings-Meta-Datei nennt `git_commit: 39c770a`, während die Evaluation `39c770a-dirty` meldet. Außerdem ist `history` leer und es werden keine Hashes der Trainingsquelle oder sämtlicher Feature-/Trainingsdateien gespeichert. Die vorhandenen Tensorvergleiche und Eval-Hashes reichen aus, um diesen konkreten Vergleich plausibel zu verifizieren, aber nicht für eine lückenlose spätere Rekonstruktion allein aus den Metadaten.

## Reward-, Feature- und Laufzeitprüfung

`KILLED_OPPONENT` erhält im Training +5. `GOT_KILLED` erhält -5 und `KILLED_SELF` 0; damit wird ein eigener Tod trotz beider Events genau einmal mit -5 bewertet. Das entspricht der im Repository dokumentierten Event-Semantik. Der offizielle Spielscore verwendet ebenfalls +5 je Kill, wie sowohl Metadaten als auch Rohscoreidentität bestätigen.

Der elfte Kanal markiert Gegnerpositionen. Die Legal-Maske blockiert Gegnerfelder für Bewegungen, und die Fluchtfeldsuche behandelt Gegnerpositionen ebenfalls als blockiert. Diese Semantik ist konsistent und verhindert offensichtlich ungültige Schritte. Sie ist jedoch konservativ: Gegner sind dynamisch, sodass ein aktuell belegtes Feld während einer mehrschrittigen Flucht frei werden könnte; umgekehrt modelliert die Fluchtberechnung keine zukünftige Gegnerbewegung. Das ist eine Feature-Näherung, kein festgestellter Evaluationsfehler.

Die Inferenz läuft ausdrücklich auf CPU. Die maximalen gemessenen Denkzeiten sind 8.51 ms für die Baseline und 27.27 ms für den Kandidaten, ohne Timeout-Überschreitungen, weit unter 500 ms. Modellpfade sind relativ zum Agentenordner und es gibt keine absoluten Pfade im geprüften Callback. Der Entwicklungsagent ist dennoch nicht submission-reif eingefroren: Die Standardvariante `auto` wählt abhängig von vorhandenen Dateien, und Modellname/Existenz hängen von Umgebungsvariablen für Seed und Episodenzahl ab. Für eine Abgabe wäre ein eigenständiger Frozen-Agent mit festem Modell und Isolationstest nötig.

## Welche Behauptungen halten stand?

1. **„Die konkrete trainierte Policy war in diesen zwei 1000-Runden-Läufen besser als die Baseline.“ – bestätigt.** Der Scoreeffekt ist groß, nicht fragil, durch Kills erklärbar und begleitet von besserer Sicherheit. Modellverwechslung, falsche Scorearithmetik und unvollständige Läufe wurden ausgeschlossen.

2. **„Das Training beziehungsweise der Gegnerkanal verursacht genau diesen Effekt.“ – nicht isoliert belegt.** Zwischen Baseline und Kandidat liegen 5000 Updates unter der vollständigen Task-3-Konfiguration. Der Gegnerkanal ist in der Baseline zwar nullinitialisiert und wird danach gelernt, aber gleichzeitig ändern sich alle Modellgewichte durch Replay, Reward und Optimierung. Es gibt keine Ablation mit weiterhin nullgehaltenem Gegnerkanal und keinen Training-ohne-Killreward-Kontrollarm. Der Effekt darf als Ergebnis der gesamten Task-3-Trainingsprozedur, nicht als isolierter Kanal- oder Rewardeffekt bezeichnet werden.

3. **„Der Effekt ist robust über Trainingsseeds.“ – nicht belegt.** Es existiert nur Trainingsseed 11. AGENTS.md verlangt auf Rung 3 mehrere Trainingsseeds. Gerade DQN-Fine-Tuning kann stark seedabhängig sein.

4. **„Die einzelne 1000er-Evaluation misst den allgemeinen Effekt zuverlässig.“ – eingeschränkt.** Sie ist eine starke Bestätigung für diese Policy, aber wegen des ungesäten stdlib-Zufalls der Gegner keine deterministische Replikation. Laut den Projektregeln sind Single-Run-Differenzen auf Rung 3 nur Bestätigung, keine endgültige Effektschätzung. Der große Abstand von +3.366 ist wesentlich größer als die dokumentierte Gegner-Rauschordnung, was einen reinen Zufallserklärungsversuch unplausibel macht; mehrere Trainingsseeds und idealerweise wiederholte Held-out-Evaluationen bleiben dennoch erforderlich.

5. **„Das ist der globale Task-3-Beststand.“ – nicht belegt.** Geprüft wurde nur gegen die lokale 11-Kanal-Baseline und nur im `peaceful`-Feld. Es fehlen Vergleiche zu anderen Teamagenten, mehreren Trainingsseeds und zum härteren `coin_collector_agent`.

6. **„Das Modell kann jetzt eingefroren beziehungsweise eingereicht werden.“ – als Zwischenkandidat ja, als endgültige Submission nein.** Ein Freeze kann die konkrete Policy konservieren, aber wissenschaftlich fehlen Multiseed-Robustheit und härtere Gegner. Technisch sollte der Freeze die dynamische `auto`-/Umgebungslogik entfernen und eigenständig getestet werden.

## Endurteil und empfohlene Formulierung

Ich konnte den engen Policy-Claim nicht widerlegen. Die sachlich zulässige Schlussfolgerung lautet:

> Das mit Seed 11 für 5000 Episoden gegen drei `peaceful_agent` trainierte Modell erzielte in einer 1000-Runden-Bestätigung auf denselben Arenaseeds einen um 3.366 höheren mittleren Score als die 11-Kanal-Task-2-Baseline, hauptsächlich durch +0.948 Kills pro Runde; zugleich sanken Suizide um 5.4 Prozentpunkte. Die Effekte sind in `analyze.py` nicht fragil. Wegen nicht vollständig reproduzierter Gegnertrajektorien und nur eines Trainingsseeds ist dies ein starker praktischer Kandidat, aber noch kein multiseed-robuster Task-3-Endstand und keine isolierte Evidenz für den Gegnerkanal allein.

Vor einem mehrstündigen Sweep ist dieser Kandidat als Ausgangspunkt gerechtfertigt. Der nächste beweiskräftige Schritt ist kein weiteres Seed-11-Fine-Tuning, sondern derselbe vorab festgelegte Trainingsarm mit mindestens zwei weiteren Trainingsseeds und anschließender zusammengefasster Bewertung; ein Frozen-Agent darf parallel zur Beweissicherung angelegt werden, sollte aber nicht als endgültige Submission bezeichnet werden.
