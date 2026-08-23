# Benedicts Logbuch

Persönliches Arbeitstagebuch: Wo ich stehengeblieben bin, woran ich gerade arbeite,
welche Ideen ich habe und warum ich mich so entschieden habe.

---

## Einträge

### 2026-08-03 — Pipeline steht: tabellarisches Q auf 4 Wand-Bits

- **Stand:** `agent_code/benedict_coin_collector/` ist kein Template mehr, sondern ein
  funktionierender (wenn auch absichtlich dummer) Q-Learning-Agent. Training läuft durch,
  die Q-Tabelle wird gespeichert und geladen, und sie enthält das, was sie enthalten soll.
  Das war das Ziel für Stufe 1 — nicht ein guter Agent, sondern eine Infrastruktur, der ich traue.

- **Gemacht:**
  - φ als *Zustandsabstraktion* gebaut: 4 Bits „Nachbarfeld blockiert?" in der Reihenfolge
    U/R/D/L, `FEATURE_SIZES = (2,2,2,2)`, gemischt-radix zu einem Zeilenindex geflacht
    (`encode`). 16 Zeilen, davon sind **11 überhaupt erreichbar** — kein Feld in der Arena ist
    auf drei oder vier Seiten blockiert. Das habe ich vorher durchgezählt, nicht im Nachhinein
    gemerkt, und es ist später der Test, ob die Tabelle stimmt.
  - `setup`/`act` mit ε-greedy, ε = 0 wenn nicht trainiert wird (sonst wirft der Agent im
    Turnier jeden zehnten Zug weg).
  - `train.py` mit dem TD-Update aus `eq:L26:q-update`, dazu `end_of_round` **ohne**
    Bootstrap-Term.
  - Hyperparameter fürs Erste: α = 0,1 · γ = 0,9 · ε = 0,2 · Schrittkosten −0,1 ·
    `COIN_COLLECTED` +5 · `INVALID_ACTION` −1 · `WAITED` −0,1. Alles geraten, nichts optimiert.
  - Ein kleines Skript in `scratchpad/print_table.py`, das die Tabelle mit dekodierten
    Feature-Bits und bester Aktion pro Zeile ausgibt. Läuft als `uv run python -m scratchpad.print_table`
    (als Datei aufgerufen liegt nur `scratchpad/` im Pfad, nicht das Repo-Root).

- **Der Bug, den ich mir merken will:** In `game_events_occurred` stand
  `if old_game_state is not None: return` — genau verkehrt herum. Effekt: bei normalen Schritten
  wurde *nie* aktualisiert, und im ersten Schritt einer Runde lief das Update mit `s = None`
  durch. NumPy behandelt `None` als `np.newaxis`, `self.q[None, a] += …` schreibt also
  klaglos in **Zeile `a`** — der Aktionsindex als Zustandsindex. Kein Fehler, kein Absturz,
  Training „läuft durch". Gelernt wurde faktisch nur einmal pro Runde aus `end_of_round`.
  Aufgefallen ist es erst beim Ausdrucken der Tabelle. Das ist die Lehre: *dass* das Training
  läuft, sagt nichts darüber, *ob* es etwas lernt.

- **Was die Tabelle nach dem Fix zeigt:**
  - In allen 11 erreichbaren Zeilen sind die blockierten Richtungen die niedrigsten Werte —
    11 von 11, keine Ausnahme. Der Abstand legal/illegal liegt bei etwa 1,0–1,5 und spiegelt
    damit einfach die −1 für `INVALID_ACTION` wider.
  - Zeile 0 (alles frei): alle vier Züge zwischen 3,17 und 3,90, also praktisch gleich. Genau
    richtig — ohne Münzinformation *sind* die Richtungen für diesen Agenten ununterscheidbar.
    Wenn ich die Münzrichtung dazunehme, muss diese Flachheit verschwinden. Das ist ein
    schöner Vorher/Nachher-Vergleich für den Bericht.
  - `BOMB` liegt überall unter der besten Aktion, obwohl `KILLED_SELF` in meiner Reward-Tabelle
    gar nicht vorkommt. Der Druck kommt allein daher, dass die Episode endet und im
    Terminal-Update kein `γ·max Q` steht. Das ist der sichtbare Beleg dafür, dass die Sache mit
    Q(terminal, ·) = 0 keine Formalie ist.

- **Nächster Schritt:**
  - Baseline sauber messen (`tools/evaluate.py`, 300 Runden, Label). `invalid` sollte nahe 0
    sein; der `coins`-Wert ist die Zahl, die die Münzrichtung schlagen muss.
  - Dann φ um **eine** Komponente erweitern — Vorzeichen des Offsets zur nächsten Münze,
    9 Werte, `|Ŝ| = 11 × 9`. Bewusst *nicht* die BFS-Richtung: die wäre kleiner und schneller,
    ist aber auf `coin-heaven` schon fast die optimale Politik und damit nah an „Merkmal liefert
    die beste Aktion". Interessanter ist ohnehin der Vergleich beider Varianten als
    kontrolliertes Experiment.
  - Offen: Belohnung +5 für eine Münze ist gegriffen, das Spiel gibt +1. Irgendwann als
    Ablation gegenprüfen, statt es einfach stehen zu lassen.

### 2026-08-02 — Vorlesungsskript L25/L26 gelesen, Richtung für Stufe 1

- **Stand:** `agent_code/benedict_coin_collector/` existiert mit `callbacks.py` + `train.py`,
  im Wesentlichen noch Template. Ab jetzt arbeite ich stärker eigenständig — mit Ben und Maxi
  bleibe ich im Austausch, aber `KONZEPT.md` ist Maxis & Ben's Papier, nicht mein Plan. Ich nehme
  daraus, was mich überzeugt, und weiche ab, wo ich es anders sehe.

- **Gemacht:** Die RL-Vorlesungen im Skript gefunden und gelesen:
  `~/Projects/scriptify/work/MLE/L25/L25.tex` (MDP, Bellman, Value Iteration, Einführung von Q)
  und `L26/L26.tex` (Bellman-Gleichung für Q, ε-greedy/Softmax, TD-Update, Q-Learning-Algorithmus,
  k-Schritt-TD, Q als Regressionsproblem, potenzialbasiertes Shaping). Das sind die Vorlesungen
  vom 14./16.07., auf die `final_project.pdf` verweist.

- **Was ich daraus mitnehme:**
  - Der Rahmen ist weiter als gedacht. L26 S. 7 nennt als Approximatoren für Q ausdrücklich
    lineare Regression, Random Forests *und* neuronale Netze. Ich muss mich also nicht früh auf
    eine Modellklasse festlegen, um den Vorlesungsbezug zu erfüllen — die Merkmale und das
    Trainingsgerüst sind das, was ich austauschbar halten will, nicht das Modell.
  - Beim Shaping gilt γΦ(s′) − Φ(s), nicht Φ(s′) − Φ(s). Die Tafel hatte das γ weggelassen;
    ohne geht die Teleskopsumme für γ < 1 nicht auf. Bei γ ≈ 0,9–0,99 relevant. Die Rechnung
    (`eq:L26:telescope`) ist ein fertiger Absatz für den Bericht.
  - Terminalzustände auf Q = 0 — im Skript Schritt (0) des Algorithmus. Wenn im letzten Übergang
    der Bootstrap-Term stehen bleibt, lernt der Agent, dass Sterben so gut ist wie Überleben.
    Das ist die eine Stelle, an der ich mir sicher bin, dass sie stimmen muss.

- **Richtung, nicht Festlegung:** Erst mal Stufe 1 (`coin-heaven`, reine Navigation) als
  Funktionstest der ganzen Pipeline — irgendeine kleine diskrete Merkmalsdarstellung, tabellarisches
  Q, ε-greedy. Ziel ist dabei weniger der Agent als die Infrastruktur: Training läuft durch,
  Metriken werden geloggt, ich sehe eine Lernkurve. Sobald das steht, ist der Rest Variation.
  Was ich mir bewusst offen lasse: Größe und Zuschnitt der Merkmale, ob Tabelle oder gleich
  lineares Modell, wie viel Replay ich am Anfang wirklich brauche.

- **Ideen fürs Weiterdenken (noch nichts entschieden):**
  - **SARSA als zweites Modell.** Das Skript leitet Q-Learning und SARSA aus derselben Gleichung
    her; Unterschied ist ein Eintrag im gespeicherten Tupel. Gleiche Merkmale, gleiche Daten,
    ein Schalter — sehr billiges kontrolliertes Experiment, und inhaltlich interessant, weil
    ε-greedy-Q-Learning gern in die eigene Bombe läuft, wo SARSA vorsichtiger wird.
  - **k-Schritt-TD** (`eq:L26:k-step`) für Stufe 2: Bei `BOMB_TIMER = 4` kommt die Belohnung für
    eine gute Bombe vier Schritte nach der Entscheidung an. Mit k = 1 wandert die Gutschrift nur
    einen Schritt pro Episode rückwärts.
  - **Batch-Hinweise aus `rem:L26:batches`**, falls Replay nötig wird: nicht mehrere Übergänge aus
    derselben Episode, Curriculum, Gewichtung nach |TD-Fehler| (= Prioritized Replay). Steht schon
    im Vorlesungsalgorithmus drin, ist also kein Zusatzaufwand für den Bericht.
  - Offene Frage an mich selbst: Lohnt sich Verhaltensklonen vom `rule_based_agent` als Warmstart
    früher, als ich denke? Die Bombenflucht-Sequenz findet ε-greedy praktisch nie von allein.

<!-- Neueste Einträge oben. Format:
### JJJJ-MM-TT — Kurztitel
- **Stand:** wo ich aufgehört habe
- **Gemacht:** was passiert ist
- **Entscheidung + Warum:** ...
- **Nächster Schritt / Ideen:** ...
-->
