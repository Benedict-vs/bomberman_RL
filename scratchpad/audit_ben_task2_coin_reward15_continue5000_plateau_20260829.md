# Adversarialer Audit: Task-2-Coin-Reward15-Fortsetzung um 5000 Episoden

Datum: 2026-08-29

## Auftrag und Prüfhaltung

Dieser Audit versucht ausdrücklich, die Behauptung zu widerlegen, dass das Endmodell der weiteren 5000-Episoden-Fortsetzung keinen praktischen Fortschritt gegenüber dem eingefrorenen `dqn_task2_coin_reward15` bringt und dass dieselbe Trainingslinie deshalb gestoppt werden sollte. Die drei Aussagen „Endmodell nicht besser“, „kein weiterer identischer Lauf sinnvoll“ und „kein Zwischencheckpoint besser“ werden getrennt beurteilt.

## Vorläufiger Status

Prüfung läuft; dieser Bericht wird inkrementell ergänzt. Außer dieser neuen Datei werden keine Dateien verändert.

## Datenintegrität und Paarung

- Beide Evaluationsdateien enthalten genau 1000 Zeilen und 1000 eindeutige Seeds. Die Tupel `(round, seed, slot)` stimmen zeilenweise vollständig überein: Runden 0–999, Seeds 20260731–20261730, Slot 0. Szenario (`classic`), Framework-Einstellungen, Python-Version und Git-Zustand (`39c770a-dirty`) stimmen überein.
- Die CSVs sind vollständig und enthalten dieselben Spalten. Beide Metadateien deklarieren 1000 Runden und dieselbe Seedspanne. Es gibt keine Invalid Actions und keine `think_over_limit`-Ereignisse.
- `tools/analyze.py --compare ... --preset task2 --markdown` erkennt korrekt 1000 gepaarte Runden. Es meldet keine fragile Zeile. Die primären Resultate B−A sind: Score −0,018, 95%-CI [−0,200; +0,162], sign-flip p=0,8555; Suizide −0,010 [−0,022; +0,001], p=0,1250; Kisten −0,640 [−2,607; +1,383], p=0,5255; Bomben −6,177 [−7,007; −5,338], p<0,0001; Überleben +0,010 [−0,001; +0,022], p=0,1250.

Damit ist der Score des Endmodells weder nachweisbar besser noch nachweisbar schlechter. Die Fortsetzung hat nur eine klare Verhaltensverschiebung hin zu weniger Bomben gezeigt, nicht einen Gewinn im primären Task-2-Ziel.

## Rohmetriken und Aktionsrekonstruktion

Die Rohmittelwerte bestätigen die bisherige Zusammenfassung:

| Metrik | Frozen-Beststand | Fortsetzung Ende | gepaarte Änderung |
|---|---:|---:|---:|
| Score/Münzen | 7,699 | 7,681 | −0,018, nicht nachgewiesen |
| Suizide | 0,024 | 0,014 | −0,010, nicht nachgewiesen |
| Kisten | 109,835 | 109,195 | −0,640, nicht nachgewiesen |
| Bomben | 35,964 | 29,787 | −6,177, nachweisbar |
| Bewegungen | 312,855 | 292,625 | −20,230, 95%-Bootstrap-CI [−26,094; −14,258], sign-flip p<0,0001 |
| Überleben | 0,976 | 0,986 | +0,010, nicht nachgewiesen |

`WAIT` ist nicht direkt in der CSV gespeichert. Die lückenlose Rekonstruktion lautet `steps − moves − bombs − invalid`; Invalid ist in beiden Läufen stets null. Sie ergibt 40,148 vs. 68,600 WAITs pro Runde, also +28,452 [23,117; 33,836], sign-flip p<0,0001. Der Median steigt von 24 auf 31. Das Endmodell ist somit eindeutig passiver.

Runden mit mindestens einer Münze unterscheiden sich nicht (0,980 vs. 0,982; +0,002 [−0,010; +0,014]). Ebenso ist der Anteil der 400-Schritt-Runden nicht nachweisbar verschieden (0,894 vs. 0,912; +0,018 [−0,007; +0,043]). Die mittleren Schritte unterscheiden sich ebenfalls nicht belastbar. Die zusätzliche Passivität liefert also weder mehr Münzrunden noch nachgewiesene Effizienz- oder Sicherheitsgewinne.

## Provenienz, Modellidentität und Fairness der Agentenordner

- Frozen-Modell laut Meta und Datei: `dqn_task2_coin_reward15_seed11.pt`, SHA-256 `b3f2756552d015adecd4bcb1d16a02ca26b97e9a891222d0c4ce909da19671e0`.
- Fortsetzungs-Endmodell laut Meta und Datei: `ben_task2_escape_crate_wait003_coin_reward15_continue5000_from_reward15_v1_seed11.pt`, SHA-256 `d4f5faa2a3839608f49afc2e30980f1bcf38723d58c6bda3b04533ca5cce6a9b`.
- Der Episode-5000-Checkpoint hat wegen Serialisierungsdetails einen anderen Datei-SHA, ist aber Schlüssel für Schlüssel und Tensor für Tensor exakt identisch zum Endmodell. Das evaluierte Modell ist daher tatsächlich der Abschlusszustand.
- Das beim Fortsetzungstraining deklarierte Quellmodell ist `ben_task2_escape_crate_wait003_coin_reward15_finetune2000_from_wait003_v1_seed11.pt`. Dieses ist tensoridentisch zum Frozen-Modell; der Lauf startete folglich exakt von der verglichenen Policy.
- `features.py` und `model.py` der beiden Agentenordner sind byte-/diff-identisch. In `callbacks.py` unterscheiden sich Kommentar, Modellwahl und die umgebungsvariable Experiment-Konfiguration; ab `def setup` ist der gesamte Inferenzcode diff-identisch. Für die angegebenen Umgebungsvariablen wählen beide `reachable_safe_tiles`, 10 Kanäle, lineare Visit-Count-Kodierung und dieselbe Action-Maskierung. Die verschiedenen Callback-SHAs sind deshalb kein semantischer Confounder der Inferenz, sondern Folge der zusätzlichen Konfigurationshülle im Entwicklungsagenten.
- Die gemessenen Laufzeiten sind beide weit unter dem 500-ms-Limit; der unterschiedliche Maximalwert (23,0 ms vs. 8,8 ms) ist ein einzelner Laufzeitausreißer und kein Policyvorteil.

Der Einwand „unfairer Vergleich wegen unterschiedlicher Agentordner“ lässt sich damit nicht halten.

## Tatsächliche Trainingskonfiguration und Vollständigkeit

Die Trainings-CSV enthält genau Episode 1 bis 5000. Es existieren vollständig alle 50 vorgesehenen Checkpoints in 100er-Schritten von 100 bis 5000; alle 50 besitzen verschiedene Datei-Hashes. Metadaten und Code bestätigen:

- Trainingsseed 11, `classic`, Start vom Reward-1,5-Modell;
- Coin-Reward 1,5, Crate-Reward 0,3, WAIT an sicherer angrenzender Kiste −0,03, Coin-Potential deaktiviert;
- 10 Kanäle, `reachable_safe_tiles`, Visit Count, Symmetrieaugmentation und Legal-Action-Mask;
- Lernrate 1e−4, Gamma 0,99, Batch 64, Replay 200000, Minimum 5000, Soft-Target-Tau 1e−4;
- konstantes Epsilon 0,05;
- frischer Adam-Optimizer, frischer Replay-Puffer und ein beim Start aus dem Online-Netz kopiertes Target-Netz.

Es handelt sich also um eine gewichtsmäßige Fortsetzung derselben Reward-/Feature-Linie, aber nicht um eine bitgenaue Fortsetzung des alten Optimizer-, Replay- und Target-Zustands. „Weitere identische Trainingslinie“ ist als gleicher Lernaufbau vertretbar; „unterbrechungsfrei dasselbe Training“ wäre falsch.

## Trainingskurve und Plateaubehauptung

Die zehn aufeinanderfolgenden 500er-Blöcke liefern keinen monotonen Aufwärtstrend. Score pro Trainingsrunde liegt zwischen 3,304 und 3,768; der erste Block liegt bei 3,594 und der letzte bei 3,600. Kisten liegen zwischen 56,378 und 62,292; der erste Block bei 60,302 und der letzte bei 59,922. Suizide liegen durch das konstante Epsilon von 0,05 durchgehend sehr hoch zwischen 0,846 und 0,906; der erste und letzte Block sind beide 0,876. Auch Bomben und WAIT kehren am Ende ungefähr auf das Anfangsniveau zurück.

Der Loss sinkt nur geringfügig von ungefähr 0,038–0,039 auf 0,036 und ist kein greedy Policymaß. Der Replay-Puffer erreicht im vierten 500er-Block seine Kapazität und bleibt danach voll. Die Kurve stützt daher „kein sichtbarer Lernfortschritt über diese 5000 Episoden“, beweist aber mathematisch nicht, dass beliebig langes weiteres Training nie helfen kann.

## Zwischencheckpoints: entscheidende Beweisgrenze

Hier lässt sich die starke Stop-Behauptung teilweise widerlegen: Alle 50 Zwischencheckpoints sind vorhanden, aber keiner wurde in den vorliegenden Daten greedy auf den gepaarten 1000 Arenen evaluiert. Die explorativen Trainingswerte können einen greedy Checkpoint nicht zuverlässig rangieren; nach den eigenen Messregeln ist eine Trainingskurve kein Ergebnis. Einzelne 500er-Blöcke um Episode 2501–3000 haben höhere Trainingsmittelwerte (Score 3,768, Kisten 62,292, Überleben 0,154) als Anfang und Ende. Das kann reines Rauschen sein, lässt aber einen besseren Zwischencheckpoint zumindest plausibel und ungetestet.

Daher ist **nicht bewiesen**, dass kein Zwischencheckpoint besser als der Frozen-Beststand ist. Ohne neue Evaluation darf der Audit nur sagen, dass das Endmodell keinen Fortschritt zeigt. Eine billige zukünftige Sichtung könnte ausgewählte Checkpoints um den besten Trainingsblock zunächst über 100 greedy Runden screenen; für die aktuelle Entscheidung ist das optional, weil Trainingsepisoden bei Epsilon 0,05 keine belastbare Vorselektion liefern und ein solcher Screen ein neues Experiment wäre.

## Adversariales Gesamturteil

1. **„Das Endmodell ist besser“ – widerlegt bzw. nicht demonstriert.** Der vollständig gepaarte 1000er-Test zeigt Score −0,018 [−0,200; +0,162], keine nachweisbaren Gewinne bei Suiziden, Überleben, Kisten, Münzrunden oder Timeouts und gleichzeitig klar mehr WAIT, weniger Bewegung und weniger Bomben. Der Vergleich ist trotz unterschiedlicher Callback-SHAs semantisch fair. Der Frozen-Agent bleibt die sachlich richtige Auswahl.

2. **„Ein weiterer gleicher Lauf ist nicht sinnvoll“ – als praktische Entscheidung bestätigt, nicht als Naturgesetz.** 5000 zusätzliche Episoden verbrauchten viel Zeit, die Trainingskurve zeigt keinen Trend und das Endmodell verbessert keine Task-2-Zielmetrik. Angesichts des Opportunity Cost ist das Stoppen dieser unveränderten Seed-11-/Hyperparameter-Linie gut begründet. Die Daten beweisen jedoch weder, dass mehr Episoden niemals helfen, noch dass ein anderer Trainingsseed oder veränderte Optimierung nichts bringen könnte. Die präzise Formulierung lautet: *Kein weiterer unveränderter Fortsetzungslauf ist durch die vorhandene Evidenz gerechtfertigt.*

3. **„Kein Zwischencheckpoint ist besser“ – nicht geprüft und darf nicht behauptet werden.** Die vorhandene Evidenz evaluiert nur das Endmodell. Die Trainingskurve kann diese Lücke nicht schließen.

## Schluss

Der Kernschluss hält dem Widerlegungsversuch stand: Das Episode-5000-Endmodell bringt keinen nachgewiesenen praktischen Task-2-Fortschritt, zeigt ein passiveres Aktionsprofil und soll den Frozen-Beststand nicht ersetzen. Das Stoppen weiterer identischer Langläufe ist eine vernünftige ressourcenbezogene Entscheidung. Der Schluss muss aber ausdrücklich auf das **Endmodell und diese unveränderte Trainingslinie** begrenzt bleiben; über die Qualität der Zwischencheckpoints liegt kein greedy Nachweis vor.
