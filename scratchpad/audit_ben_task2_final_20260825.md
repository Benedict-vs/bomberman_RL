# Adversarialer Abschluss-Audit: `ben_task2` (2026-08-25)

## Zu widerlegende Behauptung

`visit_count_v1_7000ep_seed11` ist der derzeit am besten belegte Task-2-Stand und kann als Ausgangspunkt für Task 3 eingefroren werden.

## Prüfungen und Befunde

- **Aktives Modell:** `callbacks.py` lädt `ben_task2_visit_count_v1_7000ep_seed11.pt` mit `linear_10`. Die getrennte 1.000-Runden-Evaluationskopie ist bytegleich mit diesem Modell.
- **Messung:** Visit Count schlägt die neun-kanalige Nullkontrolle beim Score gepaart um `+0,164` mit 95-%-KI `[+0,128, +0,200]`; das Ergebnis ist nicht als fragil markiert. Gegenüber späteren Armen bleibt `0,285` der höchste belegte Task-2-Score.
- **Paarung:** Beide maßgeblichen Evaluationen verwenden 1.000 Runden, Szenario `classic` und Seeds `20260731..20261730`. Die ersten 100 Runden der eingefrorenen Visit-Count-Kopie reproduzieren die historische 100-Runden-Messung in allen Spielmetriken exakt.
- **Rewardsemantik:** `GOT_KILLED=-5` und `KILLED_SELF=0`; ein Suizid wird dadurch trotz beider Frameworkevents genau einmal bestraft. Kisten erhalten `+0,2`, Münzen `+1`, der allgemeine Schrittmalus ist `-0,05`.
- **Explosionsgeometrie:** Featurecode und unverändertes Framework stoppen an Steinwänden, aber nicht an Kisten. Das entspricht dem tatsächlich ausgeführten `items.Bomb.get_blast_coords`; eine abweichende Annahme, Kisten müssten den Strahl stoppen, wäre für dieses Framework falsch.
- **Submission-Unabhängigkeit:** Inferenz verwendet relative Pfade und importiert nichts aus `tools/` oder Task 1. Der defensive `tools.trainlog`-Import liegt nur in `train.py` und fällt bei der Submission sauber auf `None` zurück.
- **CPU-Budget:** Über 1.000 Runden beträgt der Mittelwert der rundenweisen mittleren Denkzeit `0,145 ms`, das 99-%-Quantil der rundenweisen Maxima `0,349 ms`, das beobachtete Maximum `11,255 ms`; kein Schritt überschreitet `500 ms`.
- **Tests:** 60 Tests laufen erfolgreich.

## Gefundene Einschränkungen

1. **Nur ein Trainingsseed:** Der konkrete Modellvergleich ist stark, belegt aber keine Robustheit des Trainingsergebnisses über mehrere Initialisierungen. Die Mehrseed-Pflicht beginnt laut Projektkonvention erst ab Task 3; für spätere Aussagen muss sie eingehalten werden.
2. **`train.py` passt nicht zum aktiven Modell:** Es beschreibt absichtlich noch den abgeschlossenen negativen `safe_coin_wait002`-Arm. Ein versehentliches Training von `ben_task2` würde daher nicht den aktiven Beststand reproduzieren und könnte historische Ziele treffen.
3. **Ordner nicht submissionsreif:** `agent_code/ben_task2` enthält 13 Modelle, Live-PNGs, Tests, Bytecode und Entwicklungslogs. Das beeinflusst die gemessene Inferenz nicht, ist aber kein sauber eingefrorener Task-2-Agent.
4. **Task-2-Qualität bleibt begrenzt:** Score `0,285`, 763 Null-Münz-Runden und 984 Timeouts zeigen, dass der Beststand relativ zu den getesteten Armen gilt, nicht dass Task 2 gelöst ist.

## Urteil

Der Audit widerlegt die Beststandsbehauptung nicht: `visit_count_v1` ist der am besten belegte vorhandene Task-2-Arm und die Visit-Count-Information hat gegenüber der fairen Nullkontrolle einen robusten Scoreeffekt. Er widerlegt aber die stärkere Behauptung, der Entwicklungsordner sei bereits einfrier- oder submissionsfertig. Vor Task 3 sollte eine neue, kollisionsfreie, schlanke Baseline-Kopie erstellt werden, deren Inferenz eindeutig auf das bytegleiche Beststandsmodell zeigt und deren Trainingsstatus ausdrücklich eingefroren ist; vorhandene Evidenzdateien dürfen dabei nicht verschoben oder gelöscht werden.

## Nachprüfung der frozen Baseline

`agent_code/dqn_task2` wurde anschließend als eigenständiger Inferenzagent mit bytegleichem Modell angelegt. Über die ersten 100 festen Evaluationsseeds stimmen alle Spielmetriken zeilenweise exakt mit der bestätigten 1.000-Runden-Visit-Count-Datei überein; die zuvor gefundene Übergabelücke ist damit für die Inferenz geschlossen.
