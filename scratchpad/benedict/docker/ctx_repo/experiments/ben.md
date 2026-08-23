## Modellauswahl und Funktionsweise

Als Basismodell wurde **tabellarisches Q-Learning** gewählt, da es für die klar abgegrenzte Aufgabe des Münzensammelns eine einfache, interpretierbare und ressourcenschonende Methode darstellt. Jeder Spielzustand wird durch `state_to_features` in eine kompakte, diskrete Repräsentation überführt. Mithilfe einer **Breitensuche (Breadth-First Search, BFS)** wird auf dem ungewichteten Spielfeld der kürzeste begehbare Weg zur nächsten erreichbaren Münze bestimmt. Die daraus gewonnene Richtungsinformation wird in die Zustandsmerkmale aufgenommen. Die BFS gibt dabei nicht unmittelbar die auszuführende Aktion vor, sondern stellt dem Lernalgorithmus lediglich räumliche Informationen zur Verfügung. Welche Aktion in einem Zustand langfristig vorteilhaft ist, muss der Agent weiterhin anhand seiner Erfahrungen lernen.

Für jede Kombination aus Zustand \(s\) und Aktion \(a\) speichert die Q-Tabelle den erwarteten langfristigen Nutzen \(Q(s,a)\). Nach jedem Zustandsübergang wird dieser Wert entsprechend der Q-Learning-Regel aktualisiert:

$$
Q(s,a) \leftarrow Q(s,a) +
\alpha \left[r + \gamma \max_{a'}Q(s',a') - Q(s,a)\right].
$$

Die Lernrate \(\alpha = 0{,}15\) steuert, wie stark neue Erfahrungen die bestehenden Q-Werte verändern. Der Diskontierungsfaktor \(\gamma = 0{,}95\) sorgt dafür, dass neben unmittelbaren Belohnungen auch zukünftige Münzfunde berücksichtigt werden. Eine **\(\varepsilon\)-greedy Strategie** ermöglicht zu Beginn des Trainings eine breite Exploration. Der Explorationswert \(\varepsilon\) startet bei \(1{,}0\) und wird nach jeder Episode mit dem Faktor \(0{,}9995\) multipliziert, bis der Mindestwert von \(0{,}05\) erreicht ist. Dadurch nutzt der Agent im Laufe des Trainings zunehmend die bereits erlernte Strategie, probiert aber weiterhin gelegentlich alternative Aktionen aus.

Das Reward-Signal orientiert sich unmittelbar am Aufgabenziel: Für das Einsammeln einer Münze erhält der Agent eine Belohnung von \(+1\), während eine ungültige Aktion mit \(-0.1\) und Warten mit \(-0.02\) bestraft wird. Zusätzlich wird in jedem Spielschritt eine Strafe von \(-0.01\) vergeben, um kurze und effiziente Wege zu den Münzen zu fördern. Am Ende jeder Runde wird die erlernte Q-Tabelle gespeichert. Dadurch kann das Training über mehrere Durchläufe hinweg fortgesetzt und das trainierte Modell später ohne erneutes Lernen eingesetzt werden.

Um den Coincollector zu trainieren wählen wir den modus Coin-Heaven (Eine 17x17 Spielfläche ohne crates und mit 50 random verteilten Coins). Nach dem ersten trainingsdurchlauf vom ben_coin_collector (mit BFS) habe wir eine avaluation gemacht. Dabei ist uns aufgefallen, dass durchnittlich 381.51/400 schritten invalid actions sind. Wir nehmen an, dass der Agent während des Trainings durchaus gelernt hat, aber er bleibt bei der rein greedy Evaluation in ungültigen Aktionen hängen. Um dem entgegenzuwirken versuchen wir nun ungültige Aktionen stärker zu bestrafen. Also anstelle von -0.1 haben wir nun -1.0 für eine invalit action gewählt. Dadurch erhoffen wir uns, dass der agent in zukunft von invalit actions abweicht und nicht mehr hängen bleibt. Beim vergleich mit dem agenten mit -0.1 ist nun tatsächlich eine verbesserung zu sehen. Von durchnittlich 34 coins pro runde auf 43 coins und von durchschnittlichen 335/400 Stept bis Rundenabbruch auf 183 Steps. Nach weiterem analysieren auf gleichen 300 Arenen wird auch klar warum wir 335 und 183 durchschnittliche Steps gemacht haben: Coin-heaven endet entweder nach 400 Steps oder wenn es keine münzen mehr gibt. Daraus folgt, dass wir für durchschnittlich 335/400 Schritten ungefähr 71/300 mal alle Münzen gesammelt haben. Nach Implementieren von dem +5 reward für das einsammenln von Münzen kommen wir mit 18/400 Stept auf 236/300 erfolgreich abgeschlossene Runden. Somit ist mit der Rewardänderung von +1 auf 5+ eine deutliche verbesserung zu sehen.
Allerdings kommt es immer noch zu Bewegungsschleifen, wodurch nicht alle 300 Runden abgeschlossen werden. Um dies zu optimieren wählen wir nun eine weitere heransgehensweise: BFS-basiertes Potential Shaping. Hierbei wird die BFS-Distanz zur nächsten erreichbaren Münze als Zustandspotential verwendet.
Dabei bleiben Zustandsdarstellung, Lernrate, Diskontierungsfaktor, Explorationsplan, Trainingsdauer und die bisherigen Rewards unverändert.
Den Zustand definieren wir wie folgt:
Für einen Zustand `s` mit der kürzesten BFS-Distanz `d(s)` zur nächsten Münze wird das Potential definiert als:

> **Φ(s) = 1 / (d(s) + 1)**

Je kleiner die Entfernung zur nächsten Münze ist, desto größer ist das Potential.

Der zusätzliche Shaping-Reward für einen Übergang vom alten Zustand `s` zum neuen Zustand `s′` wird berechnet als:

> **F(s, s′) = η · (γ · Φ(s′) − Φ(s))**

Dabei ist:

- `γ = 0,95` der bereits verwendete Diskontierungsfaktor,
- `η = 1,0` die Stärke des Potential Shapings,
- `Φ(s)` das Potential des alten Zustands,
- `Φ(s′)` das Potential des neuen Zustands.

Das Potential hängt ausschließlich von den Zuständen vor und nach der Aktion ab. Es schreibt dem Agenten keine konkrete Aktion vor. Die Q-Tabelle muss weiterhin lernen, welche Aktionen zu Zuständen mit höherem langfristigem Wert führen.

Ziel:
Das Potential Shaping sollte dem Agenten bei jedem Übergang zusätzliche Information darüber geben, ob er sich einer Münze nähert oder von ihr entfernt. Dadurch sollten Bewegungsschleifen seltener auftreten. Erwartet werden eine höhere durchschnittliche Münzzahl, mehr vollständig abgeschlossene Runden und eine geringere mittlere Rundendauer. Die Zahl ungültiger Aktionen sollte weiterhin nahe null bleiben.


**Ergebnis:**  
Das Potential Shaping verschlechterte die Leistung deutlich. Die durchschnittliche Münzzahl sank von `42,803` auf `3,030`. Die Verschlechterung ist damit statistisch eindeutig.

Alle 300 Evaluationsrunden wird das Limit von 400 Schritten erreicht und keine Runde wurde durch das Einsammeln aller 50 Münzen beendet. Die ungültigen Aktionen stiegen von `0,00` auf `0,14` pro Runde. Dieser Anstieg ist statistisch nachweisbar, praktisch aber sehr klein. Der Agent lief daher also nicht gegen Wände, sondern blieb in gültigen Bewegungs- oder Warteschleifen hängen.

In den letzten 100 Trainingsepisoden sammelte der Agent noch durchschnittlich `46,65` Münzen. Während des Trainings halfen die verbleibenden 5 % Exploration offenbar dabei, Schleifen zu verlassen. In der vollständig greedy Evaluation fehlten diese zufälligen Auswege, wodurch die Policy kollabierte.

**Interpretation:**  
Eine mögliche Ursache ist Zustands-Aliasing. Das Potential verwendet die genaue BFS-Distanz, während die Q-Tabelle lediglich `coin_dir` und die vier lokalen Hindernisse speichert. Zustände mit unterschiedlichen Entfernungen können deshalb dasselbe Featuretupel besitzen, obwohl sie unterschiedliche Shaping-Rewards erzeugen. Dadurch können instabile oder widersprüchliche Q-Werte entstehen. Diese Erklärung ist eine Hypothese und wurde durch Versuch 4 allein noch nicht abschließend nachgewiesen.


**Fragestellung:**  
War das schlechte Ergebnis von Versuch 4 auf eine zu hohe Stärke des Potential Shapings zurückzuführen?

**Änderung gegenüber Versuch 4:**  
Als einzige Änderung wird die Shaping-Stärke von `η = 1,0` auf `η = 0,1` reduziert und es wird nochmal trainiert.

### Ergebnis von Versuch 5

Die Reduktion der Shaping-Stärke von `η = 1,0` auf `η = 0,1` verbesserte die durchschnittliche Münzzahl gegenüber Versuch 4 von `3,030` auf `6,053`. Gleichzeitig stiegen die ungültigen Aktionen jedoch von `0,14` auf `49,73` pro Runde. Alle Evaluationsrunden erreichten weiterhin das Limit von 400 Schritten.

Gegenüber der besten ungeshapten Variante aus Versuch 3 blieb Versuch 5 deutlich unterlegen. Die Münzzahl sank gepaart um `36,750` mit einem 95%-Konfidenzintervall von `[−38,727; −34,720]`. Die ungültigen Aktionen stiegen um `49,733`.

Die Hypothese, dass eine niedrigere Shaping-Stärke die ungeshapte Variante verbessert, wird daher für das konkret trainierte Modell nicht bestätigt. Eine mögliche Ursache ist Zustands-Aliasing, da das Potential die genaue BFS-Distanz verwendet, während die Q-Tabelle diese Distanz nicht als Feature enthält. Zusätzlich wurde die Zufälligkeit des Trainings nicht fest gesetzt; die Ergebnisse quantifizieren daher die Evaluationsunsicherheit des jeweiligen Modells, nicht die Varianz zwischen unabhängigen Trainingsläufen.

### Versuch 6 — Potential Shaping mit stärkeren Event-Rewards

In Versuch 6 wird untersucht, ob sich die instabile Potential-Shaping-Variante durch eine stärkere Gewichtung der zentralen Events stabilisieren lässt. Gegenüber Versuch 5 wird der Münzreward von `+5` auf `+15` und die Strafe für ungültige Aktionen von `−1` auf `−10` erhöht. Die Potential-Stärke bleibt unverändert bei `η = 0,1`.

Die stärkere Invalid-Action-Strafe soll Wandfallen verhindern, während der höhere Münzreward das Einsammeln von Münzen stärker gegenüber Bewegungsschleifen gewichtet. Erwartet werden weniger ungültige Aktionen und eine höhere Münzausbeute als in Versuch 5.

Da zwei Rewardparameter gleichzeitig verändert werden, handelt es sich um einen kombinierten Rescue-Versuch. Eine mögliche Verbesserung kann daher nicht eindeutig einem einzelnen Parameter zugeschrieben werden.

### Ergebnis von Versuch 6

Die Kombination aus Potential Shaping mit `η = 0,1`, einem Münzreward von `+15` und einer Invalid-Action-Strafe von `−10` stabilisierte die Policy gegenüber den vorherigen Shaping-Versuchen. Der Agent sammelte durchschnittlich `36,293` Münzen und führte keine ungültigen Aktionen aus. Insgesamt wurden 107 von 300 Runden vollständig abgeschlossen.

Die Variante blieb jedoch signifikant schlechter als die ungeshapte Variante aus Versuch 3. Die gepaarte Münzdifferenz betrug `−6,510` mit einem 95%-Konfidenzintervall von `[−9,267; −3,717]`. Außerdem benötigte Versuch 6 durchschnittlich `119,157` zusätzliche Schritte. Dies ist für Task 1 eine Verschlechterung, da nur 107 statt 236 Runden erfolgreich vor dem 400-Schritte-Limit beendet wurden.

Da Münzreward und Invalid-Action-Strafe gleichzeitig verändert wurden, kann die Stabilisierung nicht eindeutig einem einzelnen Parameter zugeschrieben werden. Das getestete Potential Shaping übertraf die beste ungeshapte Variante in keiner Konfiguration. Versuch 3 bleibt daher das ausgewählte tabellarische Modell.

### Versuch 7 — Ungeshapter Münzreward von +15

In Versuch 7 wird isoliert untersucht, ob ein höherer Münzreward die bisher beste tabellarische Variante aus Versuch 3 weiter verbessert. Gegenüber Versuch 3 wird ausschließlich der Reward für `COIN_COLLECTED` von `+5` auf `+15` erhöht. Die Strafe für ungültige Aktionen bleibt bei `−1`; das Potential Shaping ist mit `POTENTIAL_SCALE = 0` vollständig deaktiviert.

Es wird erwartet, dass der höhere Münzreward die zuverlässige Suche nach den verbleibenden Münzen verstärkt. Gleichzeitig könnten die größeren Q-Werte das Lernen instabiler machen. Die Variante gilt nur dann als Verbesserung, wenn die gepaarte 95%-Konfidenzgrenze der Münzdifferenz gegenüber Versuch 3 vollständig über null liegt.

### Ergebnis von Versuch 7

Die isolierte Erhöhung des Münzrewards von `+5` auf `+15` verbesserte die greedy Policy nicht. Die durchschnittliche Münzzahl sank von `42,803` auf `8,583`. Die gepaarte Differenz betrug `−34,220` Münzen mit einem 95%-Konfidenzintervall von `[−36,153; −32,197]`. Die Verschlechterung ist damit statistisch eindeutig.

Alle 300 Evaluationsrunden erreichten das Limit von 400 Schritten. Ungültige Aktionen blieben mit durchschnittlich `0,053` selten. Der Agent scheiterte daher hauptsächlich durch gültige Bewegungs- oder Warteschleifen.

In den letzten 100 Trainingsepisoden sammelte der Agent noch durchschnittlich `47,84` Münzen. Die verbleibenden 5 % Exploration halfen während des Trainings offenbar dabei, Schleifen zu verlassen. In der vollständig greedy Evaluation fehlten diese zufälligen Auswege.

Das konkret trainierte Modell mit Münzreward `+15` wird verworfen. Aufgrund der nicht fest gesetzten Trainingszufälligkeit gilt diese Aussage zunächst für das untersuchte Modell; mehrere unabhängige Trainingsläufe wären nötig, um die Trainingsvarianz vollständig zu quantifizieren. Versuch 3 mit Münzreward `+5` bleibt die beste tabellarische Variante.

# CNN-DQN für Task 1: Münzsammeln in `coin-heaven`

## Motivation für den DQN-Ansatz

Neben dem tabellarischen Q-Learning entwickeln wir einen separaten Deep-Q-Network-Agenten. Der tabellarische Agent arbeitet mit stark komprimierten, von Hand entworfenen Merkmalen, beispielsweise einer berechneten Richtung zur nächsten Münze. Dadurch kann er schnell lernen, seine Leistung hängt jedoch unmittelbar von der Qualität dieser Merkmale ab.

Der DQN verfolgt einen anderen Ansatz: Er erhält eine räumliche Darstellung des Spielbretts und soll relevante Zusammenhänge selbst lernen. Damit untersuchen wir, ob ein kleines Convolutional Neural Network aus der Position von Wänden, Münzen und dem Agenten eine Navigationsstrategie ableiten kann. Der Vergleich ist auch methodisch interessant, weil der DQN weniger vorverarbeitete Informationen erhält und deutlich datenintensiver als das tabellarische Modell ist.

Langfristig ist eine rohe Darstellung mit sieben Kanälen geplant:

1. Steinwände
2. Kisten
3. Münzen
4. eigene Position
5. Gegner
6. Bomben
7. Explosionen

Für Task 1 verwenden wir zunächst nur drei Kanäle, weil `coin-heaven` keine Kisten, Gegner, Bomben oder Explosionen enthält. Leere Kanäle würden in dieser Stufe keine zusätzliche Information liefern. Die Eingabe des ersten Modells hat deshalb die Form `3 × 17 × 17`:

1. Steinwände
2. sichtbare Münzen
3. eigene Position

Das Framework speichert Koordinaten als `(x, y)`, während das CNN Tensoren in der Reihenfolge `(Kanal, Höhe, Breite)` verarbeitet. Die Zustandsdarstellung wird daher in die Form `[Kanal, y, x]` überführt.

## Aktionsraum

Der Agent verwendet in Task 1 fünf Aktionen:

- `UP`
- `RIGHT`
- `DOWN`
- `LEFT`
- `WAIT`

Die Aktion `BOMB` ist ausgeschlossen, da Bomben in `coin-heaven` keinen Nutzen haben und ohne Gefahren- und Fluchtmerkmale nicht sicher lernbar wären.

`WAIT` bleibt bewusst Teil des Aktionsraums. Die Aktion erhält keine eigene negative Belohnung. Der Agent soll allein aus der allgemeinen Schrittstrafe lernen, dass Warten den Zustand nicht verbessert und den Weg zur nächsten Münze verlängert.

## Netzwerkarchitektur

Der erste DQN verwendet ein kleines CNN mit folgender Architektur:

```text
Eingabe: 3 × 17 × 17
→ Conv2d: 3 → 16 Kanäle, Kernel 3, Stride 1
→ ReLU
→ Conv2d: 16 → 32 Kanäle, Kernel 3, Stride 2
→ ReLU
→ Conv2d: 32 → 32 Kanäle, Kernel 3, Stride 2
→ ReLU
→ Flatten: 32 × 5 × 5 = 800 Werte
→ Linear: 800 → 128
→ ReLU
→ Linear: 128 → 5 Q-Werte


# DQN v1 – Pilotlauf mit 100 Episoden

## Ziel

Der Pilotlauf sollte prüfen, ob die DQN-Pipeline funktioniert und das CNN erste Navigationsmuster lernt. Der Agent erhielt drei `17 × 17`-Kanäle für Wände, Münzen und die eigene Position. Der Aktionsraum bestand aus vier Bewegungen und `WAIT`; `BOMB` war ausgeschlossen.

Trainiert wurde mit Replay Buffer, Target Network, Huber Loss und ε-greedy Exploration. Die wichtigsten Einstellungen waren:

- γ: 0,99
- Lernrate: 0,0001
- Replay Buffer: 20.000
- Batchgröße: 64
- Epsilon: 1,0 → 0,05 über 100.000 Schritte
- Schritt-Reward: −0,05
- Münze: +1
- ungültige Aktion: −0,5
- keine zusätzliche `WAITED`-Strafe

## Ergebnis

Nach 100 Episoden lag Epsilon noch bei ungefähr 0,62. Während des Trainings stieg die Münzzahl und die Anzahl ungültiger Aktionen sank. Der Loss blieb stabil.

In der greedy Evaluation über 300 Runden sammelte das Modell durchschnittlich 4,927 Münzen. Die Completion Rate lag bei 0 %. Der Pilot zeigte somit, dass die Pipeline funktioniert, 100 Episoden aber wahrscheinlich nicht für eine zuverlässige Policy ausreichen.


# DQN v1 – Training mit 300 Episoden

## Fragestellung

Es wurde untersucht, ob die geringe Leistung des Pilotmodells hauptsächlich durch die kurze Trainingsdauer verursacht wurde. Gegenüber dem Pilotlauf wurde nur die Trainingsdauer von 100 auf 300 Episoden erhöht.

## Ergebnis

Während des Trainings stieg die Münzleistung zunächst an, während ungültige Aktionen zurückgingen. Epsilon erreichte nach ungefähr 250 Runden seinen Endwert von 0,05. Keine Trainingsrunde wurde vollständig abgeschlossen.

In der greedy Evaluation auf denselben 300 Arenen sammelte das neue Modell durchschnittlich 6,763 statt 4,927 Münzen. Die gepaarte Verbesserung betrug +1,837 Münzen mit einem 95-%-KI von [+0,953; +2,720] und war damit statistisch nachgewiesen.

Gleichzeitig stiegen die ungültigen Aktionen von 23,40 auf 103,26 pro Runde. Die Verschlechterung von +79,867 mit einem 95-%-KI von [+59,576; +100,661] war ebenfalls signifikant. In 23 von 300 Runden führte der Agent ausschließlich ungültige Aktionen aus.

## Fazit

Mehr Training verbesserte die Münzleistung, löste aber die Zustandsfallen nicht. Als nächster Versuch wird deshalb eine Legal-Action-Maske eingeführt, die Bewegungen in Wände ausschließt. Alle anderen Einstellungen bleiben unverändert.

## DQN v3 – Erhöhter Münz-Reward

Es wurde geprüft, ob eine Erhöhung des Münz-Rewards von +1 auf +2 die häufigen `WAIT`-Aktionen reduziert. Alle anderen Einstellungen blieben gegenüber v2 unverändert.

Nach 300 Episoden sank die greedy Münzleistung von 9,433 auf 7,847. Die gepaarte Differenz betrug −1,587 Münzen mit einem 95-%-KI von [−2,873; −0,313]. Gleichzeitig stiegen die Invalid Actions signifikant von 2,64 auf 49,61.

Die Änderung verschlechterte damit sowohl Primär- als auch Diagnosemetrik und wird verworfen. Für weitere Experimente bleibt der Münz-Reward bei +1.


## DQN v4 – Höhere Schrittstrafe

Die Schrittstrafe wurde bei ansonsten identischer v2-Konfiguration von −0,05 auf −0,10 erhöht. Ziel war, Zeitverschwendung und `WAIT` zu reduzieren.

Nach 300 Episoden sank die greedy Münzleistung signifikant von 9,433 auf 7,377. Die gepaarte Differenz betrug −2,057 Münzen mit einem 95-%-KI von [−3,313; −0,850]. Für Invalid Actions wurde keine Verbesserung nachgewiesen.

Die Änderung wird verworfen. Die höhere Schrittstrafe bestraft auch sinnvolle längere Wege und löste die Warteproblematik nicht.

## DQN v5 – Legal-Action-Maske

Die Legal-Action-Maske schloss Bewegungen in Wände bei Exploration, greedy Auswahl und im Bellman-Target aus. `WAIT` blieb legal.

Nach 300 Episoden sanken Invalid Actions auf null. Gleichzeitig fiel die greedy Münzleistung signifikant von 9,433 auf 3,973. Die Differenz betrug −5,460 Münzen mit einem 95-%-KI von [−6,693; −4,267].

Die Maske verhinderte Wand-Schleifen, verschob das Problem jedoch zu häufigem Warten. Die Variante mit Coin-Reward +1 wird bei diesem Trainingsbudget verworfen.

## DQN v5/v6 – Legal-Action-Maske

Die Maske reduzierte Invalid Actions auf null, führte aber zu häufigem Warten und deutlich weniger Münzen. Unter der Maske verbesserte Coin-Reward +2 die Leistung signifikant von 3,973 auf 5,017 Münzen. Eine Verlängerung auf 1.000 Episoden zeigte keinen weiteren Effekt.

Trotzdem blieben beide Maskenvarianten deutlich unter der maskenlosen v2. Die Legal-Action-Maske wird deshalb für Task 1 verworfen.


## DQN v7 – Legal-Action-Maske mit Coin-Reward +5

Coin +5 verbesserte die Maskenvariante signifikant gegenüber Coin +2. Gegenüber der maskenlosen v2 blieb v7 jedoch signifikant schlechter: Die Münzleistung sank von 9,433 auf 5,770, mit einer Differenz von −3,663 und einem 95-%-KI von [−4,880; −2,480].

Die Legal-Action-Maske verhindert Invalid Actions, führt aber zu häufigem `WAIT`. Die Maskenfamilie wird deshalb für Task 1 verworfen.

## Optimierung des CNN-DQN

Alle Varianten wurden nach 300 Trainings­episoden auf denselben 300 `coin-heaven`-Arenen ohne Exploration evaluiert. Eine Änderung galt nur dann als verbessert, wenn das gepaarte 95-%-Konfidenzintervall 0 ausschloss.

Die Lernrate `1e-4`, `γ = 0.99`, Batch-Größe 64 und ein Epsilon-Abbau über 100.000 Schritte erwiesen sich als beste Grundeinstellungen. Den größten Fortschritt brachte ein selteneres Update des Target Networks: Mit einem Intervall von 20.000 statt 1.000 Optimierungsschritten stieg die Leistung von 9,43 auf 20,42 Münzen. Dieser Vorteil zeigte sich auch unter einem zweiten Trainingsseed.

Eine Vergrößerung des Replay Buffers verbesserte den Mittelwert weiter. Mit 200.000 Plätzen wurden in zwei Läufen 26,99 und 29,02 Münzen erreicht. Ein Warm-up von 5.000 Übergängen erzielte schließlich den bisherigen Bestwert von 29,31 Münzen. Der Vorteil gegenüber einem Warm-up von 1.000 war jedoch knapp nicht statistisch nachgewiesen. Ein Warm-up von 10.000 war mit 25,71 Münzen signifikant schlechter.

Die bisher beste Arbeitskonfiguration lautet:

- Replay Buffer: 200.000
- Batch-Größe: 64
- Warm-up: 5.000 Übergänge
- Target-Update-Intervall: 20.000
- Lernrate: `1e-4`
- Discount-Faktor: `0.99`
- Epsilon-Abbau: 100.000 Schritte

Der DQN erreicht damit rund 29 Münzen pro Runde und praktisch keine ungültigen Aktionen. Er liegt jedoch weiterhin deutlich unter dem tabellarischen Task-1-Agenten mit 50 Münzen. Als nächster Algorithmusversuch wird deshalb Double DQN untersucht, um eine Überschätzung der Q-Werte zu reduzieren.

## Weitere Optimierung des CNN-DQN

Nach der bisherigen Hyperparameteroptimierung erreichte der Standard-DQN nach 300 Trainings­episoden 29,31 Münzen. Darauf aufbauend wurden mehrere algorithmische Erweiterungen kontrolliert untersucht. Alle angegebenen Vergleiche wurden auf identischen Arenen mit gepaarten 95-%-Konfidenzintervallen durchgeführt.

### Double DQN

Zunächst wurde Double DQN implementiert. Dabei wählt das Online Network die nächste Aktion, während das Target Network diese Aktion bewertet. Dies soll die Überschätzung von Q-Werten reduzieren. Die Implementierung wurde durch einen eigenen Unit-Test abgesichert.

Double DQN erreichte jedoch nur 27,12 statt 29,31 Münzen. Die Differenz von −2,19 Münzen war knapp statistisch nachweisbar (95-%-KI: [−4,32; −0,05]). Daher wurde Double DQN verworfen und zum Standard-DQN zurückgekehrt.

### Potential-basiertes Reward Shaping

Als nächste Erweiterung wurde ein zustandsbasiertes Potential eingeführt. Dieses basiert auf der per BFS berechneten kürzesten Entfernung zur nächsten Münze:

```text
F(s, s') = β · (γ · Φ(s') − Φ(s))