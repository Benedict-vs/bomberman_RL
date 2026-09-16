# Adversarialer Audit: Task 4 Temporal Safety Full1000 (2026-09-07)

## Auftrag und Status

Dieser Bericht versucht den vorläufigen No-Go-Befund gezielt zu widerlegen. Geprüft werden Vollständigkeit, Agentenzuordnung, Modell- und Trainingsprovenienz, zulässige Paarung, Scorezerlegung, Todesmetriken, Survival, Timing, statistische Fragilität sowie die vorab in `BEN.md` festgelegten Gates.

Status: abgeschlossen.

## Kurzurteil

Der enge No-Go-Befund für **diesen konkreten Seed-11-Kandidaten nach 1.000 Trainingsepisoden** hält. Ich finde weder eine falsche Agentenzuordnung noch eine Modellverwechslung, unvollständige Daten, fehlerhafte Scorearithmetik, vertauschte Todesursachen, einen Timingverstoß oder eine Fragilitätsmarkierung, die den Kandidaten retten würde. Der Kandidat erfüllt das vorab definierte Advancement-Gate nicht: Sein Scoreeffekt ist nicht positiv nachgewiesen (`−0,121`, 95-%-CI `[−0,321,+0,078]`, Sign-Flip-`p=0,2347`) und die Suizid-Punktschätzung liegt mit `+0,065` über der Guard-Grenze `+0,03`.

Eine Präzisierung ist notwendig: Das Suizidintervall `[+0,024,+0,106]` schließt null stabil aus, aber nicht die Schwelle `+0,03`. Auf einer zweiseitigen 95-%-Beweislogik ist daher eine *positive Suizidregression* nachgewiesen, nicht ebenso streng eine Überschreitung von `+0,03`. Falls das Gate entgegen seinem Wortlaut als Hypothesentest gegen `+0,03` statt als Punktschätzungsgrenze gemeint war, wäre allein dieser Teil nicht entschieden. Das rettet den Kandidaten nicht, weil das primäre Score-Advancement unabhängig davon scheitert. Ebenso rechtfertigt ein Ein-Seed-Pilot nicht die Aussage, jede zeitabhängige Sicherheitsdarstellung sei generell schädlich; er rechtfertigt nur, diesen Kandidaten nicht weiterzubefördern und die vorregistrierte Linie hier zu stoppen.

## 1. Vorab festgelegte Gates

Die relevanten Einträge stehen vor beziehungsweise während der Messkette am Anfang von `BEN.md`:

- Quick100 war ausschließlich Liveness-, Timing- und Futility-Gate. Full1000 blieb erlaubt, wenn ein praktisch relevanter Scoregewinn von mindestens `+0,12` innerhalb der Unsicherheit lag und der Suizidpunktwert nicht über `+0,03` stieg.
- Für Full1000 verlangte Advancement einen **nicht fragilen positiven Scoreeffekt**; Suizide waren Regression Guard und Timing die Turniergrenze.
- Weitere Trainingsseeds waren ausdrücklich erst nach bestandenem Full1000-Pilot freigegeben. Bei Scheitern war 3-step DQN die vorab gereihte nächste Hypothese.

Damit ist `−0,121` nicht deshalb ein Ablehnungsgrund, weil ein negativer Effekt bewiesen wäre—das ist er nicht—sondern weil der verlangte positive Effekt nicht bewiesen ist. Die obere CI-Grenze `+0,078` liegt außerdem unter dem zuvor als praktisch relevant bezeichneten `+0,12`, obwohl diese Futility-Schwelle formal für die Quick100-Fortsetzungsentscheidung formuliert wurde.

## 2. Vollständigkeit und Agentenzuordnung

Beide Evaluationsdateien bestehen die Rohdatenprüfung:

| Prüfung | Nullkanal | Kandidat |
|---|---:|---:|
| CSV-Zeilen | 4.000 | 4.000 |
| Zielagentenzeilen (`code=ben_task4`) | 1.000 | 1.000 |
| Runden | 0–999, lückenlos | 0–999, lückenlos |
| Seeds | 20260731–20261730 | 20260731–20261730 |
| Zielagentslot | immer 0 | immer 0 |
| eindeutige `(round,seed,slot)`-Schlüssel | 4.000 | 4.000 |

Die Metadaten nennen in beiden Fällen dieselbe Aufstellung `ben_task4` plus drei `rule_based_agent`, Szenario `classic`, 1.000 Runden, Base-Seed `20260731`, identische Spielregeln und denselben Callback-Hash `069ac573…4ab6`. Arm, Dateiname und Modell unterscheiden sich konsistent:

- Nullkanal: `BM_TASK4_TRAINING_ARM=temporal_safety_zero_v1`, Modell `ben_task4_temporal_safety_zero_v1_1000ep_seed11.pt`, SHA-256 `38e7e638…c00dc2b`.
- Kandidat: `BM_TASK4_TRAINING_ARM=temporal_safety_v1`, Modell `ben_task4_temporal_safety_v1_1000ep_seed11.pt`, SHA-256 `ccbd46a2…97d9d7`.

Die aktuell vorhandenen Modellbytes ergeben exakt diese beiden Hashes. Eine Modellverwechslung zwischen den zwei Auswertungen ist damit ausgeschlossen.

## 3. Trainingsartefakte und Modellprovenienz

Beide Trainings-CSVs haben einschließlich Header 1.001 Zeilen und enthalten 1.000 Episoden. Pro Arm existieren zehn Checkpoints (Episode 100 bis 1.000). Die Episode-1000-Checkpoints haben wegen PyTorch-Serialisierung andere Datei-Hashes als die Endmodelle, aber ein direktes Laden zeigt für beide Arme identische State-Dict-Schlüssel und **bitgenau gleiche Tensoren** zwischen Endmodell und jeweiligem Episode-1000-Checkpoint.

Die Trainingsmetadaten stimmen in den protokollierten Hyperparametern überein, außer den beabsichtigten Feldern Armname, `temporal_safety_mode` (`zero` versus `enabled`) und Ausgabemodell. Beide nennen:

- Trainingsseed 11, 1.000 Episoden, zwölf Eingangskanäle;
- dieselbe Quelle `ben_task4_mixed_kill_v1_12ch_source.pt`;
- Gegner `peaceful_agent,rule_based_agent,rule_based_agent`;
- dieselben Rewards, Replay-, Optimierungs-, Masken- und Augmentierungsparameter;
- `GOT_KILLED=-5,0` und `KILLED_SELF=0,0`.

Die verschiedenen Buffer-Endstände (123.337 versus 120.610 Übergänge) folgen aus verschiedenen Episodenlängen und sind kein Konfigurationsbeweis gegen das Paar.

Provenienzgrenze: Die Evaluationen stammen aus `3dac0c5-dirty`; die Eval-Metadaten hashen `callbacks.py`, aber nicht `features.py` und `model.py`. Der aktuell vorhandene Callback hat noch exakt den protokollierten Hash, doch der damalige Dirty-Tree-Zustand der importierten Feature-/Modelldateien ist allein aus den Metadaten nicht bytegenau rekonstruierbar. Das schwächt Reproduzierbarkeit und allgemeine Kausalbehauptungen, erzeugt aber keinen plausiblen armselektiven Modelltausch: beide Läufe verwenden denselben Callback-Hash und die Evaluationsmetadaten binden jeweils den richtigen, heute verifizierbaren Modellhash.

## 4. Paarung

Die 1.000 Zielagentenschlüssel `(round,seed)` stimmen zwischen den Dateien exakt überein. Eine gepaarte Rechnung pro Arena ist deshalb zulässig und behandelt 1.000 Arenen, nicht 4.000 Agentenzeilen, als Beobachtungen.

Die Paarung ist dennoch nur eine **Arenapaarung**. Die Regelagenten verwenden neben NumPy auch nicht vollständig kontrollierte stdlib-Zufälligkeit; deshalb sind ihre Trajektorien zwischen den sequenziellen Läufen nicht vollständig gepaart. Das macht den Vergleich nicht unzulässig, sondern fügt ungepaarte Gegnernoise hinzu und verbietet die Interpretation als deterministisches Within-Arena-Counterfactual. Als konservativer Gegencheck liefert ein ungepaarter Bootstrap ähnliche Schlüsse:

- Score `−0,121`, ungefähr `[−0,334,+0,095]`;
- Suizide `+0,065`, ungefähr `[+0,023,+0,107]`;
- Survival `−0,068`, ungefähr `[−0,111,−0,025]`.

Die Schlussrichtung hängt daher nicht von einer zu starken Paarungsannahme ab.

## 5. Score und Zerlegung

Für sämtliche 8.000 CSV-Zeilen gilt exakt `score = coins + 5 × kills`; es gibt keine Arithmetikverletzung. Für den Zielagenten:

| Metrik | Nullkanal | Kandidat | Kandidat − Null |
|---|---:|---:|---:|
| Score | 3,703 | 3,582 | −0,121 |
| Coins | 2,973 | 2,822 | −0,151 |
| Kills | 0,146 | 0,152 | +0,006 |

Damit gilt exakt `−0,121 = −0,151 + 5 × 0,006`. Der Killunterschied ist nicht nachgewiesen (`[−0,027,+0,039]`); der Coinrückgang ist in Wiederholungs-Bootstraps stabil negativ (ungefähr `[−0,25,−0,05]`). Der primäre Scoreeffekt bleibt über fünf Bootstrap-Seeds offen, jeweils grob von `−0,32` bis `+0,08`, und ist mit Sign-Flip-`p=0,2347` nicht fragil, sondern klar **nicht nachgewiesen**.

Der bekannte ungefähr `±0,12` Score-Noise-Floor aus wiederholten Regelagenten-Auswertungen mahnt gegen die Aussage, der wahre Effekt sei exakt `−0,121` oder der Kandidat sei sicher scoremäßig schlechter. Er ist aber kein Advancement-Joker: Das Gate verlangt positive Evidenz, und zusätzliche Messnoise kann fehlende Evidenz nicht in positive Evidenz umdeuten.

## 6. Todesinterpretation und Survival

Die vermutete falsche Todesinterpretation liegt nicht vor.

Im Framework erhält jeder explodierte Agent `GOT_KILLED`; nur beim Tod durch die eigene Bombe wird zusätzlich `KILLED_SELF` ausgelöst und die Frameworkstatistik `suicides` erhöht. `tools/evaluate.py` berechnet deshalb `killed_by_opponent = died − suicides`. In allen 8.000 Zeilen gelten exakt:

- `died = suicides + killed_by_opponent`;
- `survived + died = 1`.

Die Evaluationsmetriken zählen somit Todesursachen, nicht Rewardevents. Es wäre falsch, `GOT_KILLED` als gegnerbedingten Tod zu lesen oder bei Suiziden `GOT_KILLED + KILLED_SELF` als zwei Tode zu zählen. Genau das geschieht hier nicht. Auch die Trainingsrewardtabelle ist korrekt: der ganze Todesmalus liegt auf `GOT_KILLED`, `KILLED_SELF` ist null, sodass ein Suizid nicht doppelt bestraft wird.

| Metrik | Nullkanal | Kandidat | Differenz (95-%-CI) |
|---|---:|---:|---:|
| Suizide | 0,319 | 0,384 | +0,065 `[+0,024,+0,106]` |
| gegnerbedingte Tode | 0,070 | 0,073 | +0,003 `[−0,020,+0,026]` |
| Survival | 0,611 | 0,543 | −0,068, ungefähr `[−0,110,−0,026]` |

Die Survivaldifferenz ist exakt das Negative der Summe beider Todesdifferenzen: `−(0,065+0,003)=−0,068`. Der Sicherheitsverlust wird daher fast vollständig durch **eigene Bomben** getragen, nicht durch Gegnerbomben. Das passt zur vorgesehenen diagnostischen Trennung.

Fragilität: Die Suizid-CI bleibt über mehrere Bootstrap-Seeds positiv und Sign-Flip-`p=0,0025` stimmt mit ihr überein; keine Fragilitätsmarkierung. Dass die untere Grenze gelegentlich `+0,023` oder `+0,024` beträgt, ändert den Nachweis `>0` nicht. Sie verhindert lediglich den strengeren zweiseitigen 95-%-Nachweis `>+0,03`.

## 7. Timing

Die mittlere rundenweise maximale Denkzeit steigt von rund `0,3 ms` auf `0,7 ms`; die gepaarte Differenz beträgt `+0,351 ms` mit stabilem Intervall ungefähr `[+0,26,+0,44]`. Das ist statistisch klar, praktisch aber ungefährlich. Es gab in beiden Armen null `think_over_limit`; die globalen Zielagentenmaxima waren `14,1041 ms` und `14,2579 ms`, weit unter `500 ms`. Timing ist kein Ablehnungs- und kein Rettungsgrund.

## 8. Adversariale Alternativerklärungen

1. **„Der Score ist nur wegen Gegnernoise −0,121.“** Möglich; ein negativer Scoreeffekt ist nicht bewiesen. Das Gate verlangt jedoch einen positiven, nicht fragilen Effekt. Auch die obere gepaarte CI-Grenze erreicht `+0,12` nicht.
2. **„Die Paarung ist unzulässig.“** Nur vollständig identische Gegnertrajektorien zu behaupten wäre unzulässig. Arenapaarung ist vorhanden; ein ungepaarter Gegencheck ändert das Urteil nicht.
3. **„Suizide sind eigentlich alle Tode.“** Falsch. Framework und Rohdaten bestätigen die disjunkte Zerlegung; `GOT_KILLED` darf nicht als Fremdtod interpretiert werden.
4. **„Das falsche Modell wurde evaluiert.“** Die beiden separaten Metadaten, Armvariablen, Modellnamen und verifizierten SHA-256 widersprechen dem. Endmodelle sind tensoridentisch zu ihren jeweiligen Episode-1000-Checkpoints.
5. **„`+0,065` beweist die Verletzung der `+0,03`-Grenze.“** Nur wenn die Grenze vorab als Punktschätzungs-Guard gemeint war. Ein zweiseitiges 95-%-Intervall schließt `+0,03` nicht aus. Diese semantische Lücke sollte im Ledger präzise formuliert werden, ändert wegen des Score-Gates aber nicht das No-Go.
6. **„Ein Seed kann den Mechanismus generell widerlegen.“** Nein. Der Versuch bewertet einen konkreten trainierten Checkpoint. Nach der Vorregistrierung reicht sein Scheitern zum Stoppen dieser Linie, nicht zum universellen Verwerfen aller Temporal-Safety-Kodierungen.

## Endverdict

**No-Go bestätigt, aber eng formulieren.** Der Kandidat darf den Nullkanal oder den Mixed-Kill-Incumbent nicht ersetzen. Der Full1000-Pilot liefert keinen nicht fragilen positiven Scoreeffekt, zeigt eine robuste Zunahme eigener Bombentode und einen entsprechend robusten Survivalrückgang. Kein Integritätsfehler hebt dieses Ergebnis auf.

Zulässige Aussage: „Für den konkreten Seed-11-Checkpoint nach 1.000 Episoden zeigte der aktivierte Temporal-Safety-Kanal gegenüber dem Nullkanal keinen Scorevorteil und erhöhte die Suizidrate; der vorab definierte Advancement-Guard wurde nicht bestanden.“

Zu starke Aussage: „Temporal Safety verschlechtert den Score sicher“ oder „jede zeitabhängige Sicherheitsrepräsentation ist widerlegt.“ Der Scoreunterschied ist nicht signifikant, Gegnerzufälligkeit bleibt nur teilweise gepaart, die Codeprovenienz des Dirty Trees ist nicht vollständig gehasht und es gibt nur einen Trainingsseed.
