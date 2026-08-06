# Maxis Logbuch

Persönliches Arbeitstagebuch: Wo ich stehengeblieben bin, woran ich gerade arbeite,
welche Ideen ich habe und warum ich mich so entschieden habe.

---

## Einträge

### 2026-08-06 — Stufe 2: Ziel-Merkmal nach Art getrennt, Besuchszähler, Logpfad-Bug

- **Stand:** `maxi_task2` hat jetzt Gefahrenkarte, Gefahrenstufe und Flucht-BFS
  (uncommittet aus der letzten Sitzung). Der Merkmalssatz ist damit
  `4 Wandbits + Ziel (Richtung × Art) + Gefahr + Flucht` → **2880 Zustände**.
  Noch **nicht trainiert**, noch **nicht gemessen** — es gibt bis jetzt keine Zahl,
  die irgendetwas über Stufe 2 aussagt.
- **Gemacht:**
  - **Auslöser war ein Absturz, kein Bug:** `python main.py play --my-agent maxi_task2`
    bricht in `setup()` mit „Q-table on disk has shape (80, 6), expected (1600, 6)" ab.
    Die Formprüfung tut genau das, wofür sie da ist — die Tabelle auf der Platte stammt
    aus dem Merkmalssatz *vor* den Gefahren-Merkmalen. Alte Zeilen sind nicht
    weiterverwendbar: das gemischtradixe `encode()` vergibt jetzt andere Indizes,
    Zeile *k* bedeutet etwas völlig anderes. Auffüllen wäre schlechter als Nullen.
  - **Ziel-Merkmal nach Art getrennt** (die seit dem 05.08. offene Frage in
    `experiments/maxi.md`): eine Stelle mit 9 Werten statt 5 — `0` nichts erreichbar,
    `1..4` **Münze** in Richtung `ACTIONS[0..3]`, `5..8` **Kiste** in Richtung
    `ACTIONS[0..3]` (`COIN_OFFSET`/`CRATE_OFFSET`). `target_direction()` liefert den
    versetzten Wert, Münzvorrang und der eine BFS-Durchlauf bleiben unverändert.
  - **Besuchszähler ausgelesen:** `self.visits` gab es schon für die Lernrate, wurde
    aber nie ausgewertet. Neu zwei Spalten im Trainingslog (`states_seen`,
    `cell_coverage`) plus alle 100 Runden eine Zeile im Agentenlog — inklusive
    **Median der Besuche über die gesehenen Zellen**, denn das ist die Zahl, die
    „schon konvergiert" von „einmal berührt" unterscheidet. Läuft ohne `tools/`.
  - **Bug gefunden und behoben:** `agents.py:305` wechselt vor `setup_training()` ins
    Agentenverzeichnis. Das relative `out_dir="results/train/task2_crates"` landete
    deshalb unter `agent_code/maxi_task2/results/…` statt im Repo-Wurzelverzeichnis —
    der Kommentar in `train.py` behauptete ausdrücklich das Gegenteil. `out_dir` leitet
    sich jetzt aus `__file__` ab; kein absoluter Pfad im Quelltext, Einreichungsregel
    bleibt gewahrt. **Das alte `q_e10_s0`-Log liegt noch am falschen Ort** — vor dem
    nächsten Lauf verschieben, sonst wird weiter dorthin angehängt.
  - Rauchtest 120 Runden: Spalten füllen sich, **136/2880 Zustände (4,7 %), 2,6 % der
    Zellen, Median 1 Besuch**. Das ist ein untrainierter Agent, der früh stirbt — also
    eine Untergrenze und *kein* Ergebnis, nur der Nachweis, dass das Messgerät geht.
- **Entscheidung + Warum:**
  - **Eine 9-wertige Stelle statt zweier Stellen (Richtung, Art).** Mit getrennter
    Art-Stelle wären `(NO_TARGET, Münze)` und `(NO_TARGET, Kiste)` dieselbe Lage in zwei
    Zeilen, die sich die Erfahrung teilen. Die Versätze vermeiden diese toten Zeilen.
  - **Jetzt trennen, nicht später.** Die Tabelle muss wegen der Gefahren-Merkmale
    ohnehin von Null neu trainiert werden — das ist der einzige Moment, an dem die
    Trennung nichts extra kostet. Sonst zahlt man ein zweites volles Training.
  - **Kein Ergebnis ohne `tools/evaluate.py` bei ε = 0.** Der Rauchtest oben ist
    ausdrücklich keins; siehe die 48,2-gegen-1,45-Erfahrung aus E01.
- **Nächster Schritt / Ideen:**
  1. Trainieren und **Abdeckung beobachten**: 80 → 2880 Zeilen ist das 36-Fache der
     Stufe-1-Tabelle. Bleibt `cell_coverage` niedrig *und* der Median bei 1–2, ist die
     Frage, ob die Zeilen unerreichbar sind (harmlos, Nullen) oder nur selten besucht
     (das ist das, was erratisches Spiel erzeugt). Danach erst mehr Episoden.
  2. `CRATE_DESTROYED` in `REWARDS` — steht als TODO in `train.py` und ist auf Stufe 2
     das eigentliche Lernsignal. Ohne das lernt der Agent nur wegzulaufen.
  3. Offene Vorhersage vor dem Lauf in `experiments/maxi.md` notieren (E10), bevor
     trainiert wird — nicht danach.
  4. Unittest für die Explosionsgeometrie steht **seit dem 31.07.** offen. Die
     Gefahrenkarte ist jetzt der Kern des Agenten; das ist die Stelle, an der ein
     stiller Fehler alles kostet und in keiner Metrik als Fehler auftaucht.

### 2026-08-05 — Stufe 1 gelöst: 50/50 Münzen, schneller als die Referenz

- **Stand:** `maxi_coin_collector` ist ein tabellarischer Q-Learner auf
  `(coin_dir, up, right, down, left)` und schlägt auf `coin-heaven` den
  `coin_collector_agent`. Stufe 1 ist abgehakt, Stufe 2 (Kisten) ist offen.
- **Gemacht:**
  - **Baselines gemessen** (E00), bevor irgendetwas trainiert wurde: die beiden
    Referenzagenten holen in *jeder* Runde alle 50 Münzen in ~125 Schritten,
    `random_agent` 1,8, `peaceful_agent` 18,7. → `results/eval/baselines/`
  - **Drei Fehler im Agenten behoben,** die das erste Training sofort zum Absturz
    gebracht haben: `self.model` war noch der Wahrscheinlichkeitsvektor aus der Vorlage
    statt einer Q-Tabelle (`ValueError`, 5-Tupel gegen 6-Array), `numpy` in `train.py`
    nie importiert, kein `None`-Schutz in `update_q_table`.
  - **Drei Trainingsläufe + zwei Kontrollläufe,** je 1000 Runden, dazu je eine
    300-Runden-Messung mit ε = 0. Protokoll in `experiments/maxi.md` (E00–E02).
  - **Abbildungen** in `results/figures/`, Skript `tools/plot_task1_versions.py`.
- **Entscheidung + Warum:**
  - **`BOMB` und `WAIT` raus aus dem Aktionsraum für Stufe 1.** v1 endete in
    **100 % der 1000 Episoden** mit `KILLED_SELF`: ohne Gefahren-Merkmal ist Weglaufen
    nicht lernbar, und auf `coin-heaven` bringt eine Bombe ohnehin nichts. Kommen auf
    Stufe 2 zurück, aber nur zusammen mit den Gefahren-Merkmalen.
  - **ε-Zerfall 0,9995 → 0,997.** 0,9995^1000 = 0,61 — der erste Lauf hat die gelernte
    Politik nie bei kleinem ε ausgeführt.
  - **Zwei Änderungen auf einmal = zwei Kontrollläufe.** v1a nur Maske, v1b nur Zerfall.
    Ergebnis: keine der beiden genügt allein (Maske behebt das Sterben, lässt aber
    78 ungültige Aktionen; Zerfall allein lässt 62 % Selbstmorde). Kostet je 90 s, ist
    aber der Unterschied zwischen „wirkt" und „wir wissen, was wirkt".
  - **Schrittkosten −0,1 aktiviert** — die eigentliche Lösung.
- **Der lehrreichste Fehler:** v2 sah im Training mit Abstand am besten aus
  (48,2 Münzen/Episode) und war in der Messung mit ε = 0 der mit Abstand schlechteste
  (**1,45**). Ursache: ohne Schrittkosten sättigen die Q-Werte auf einen gemeinsamen
  Pegel, die Spreizung zwischen den Zügen verschwindet, und eine deterministische,
  ortsblinde Politik läuft dann in absorbierende Zyklen — 119 von 300 Runden mit
  **399 ungültigen Aktionen** (verklemmt in Schritt 1 gegen eine Wand), 156 Runden mit
  exakt 0 (Pendeln zwischen zwei Feldern). Im Training verdeckt ε = 0,05 das komplett.
  **Merksatz: eine Trainingskurve ist kein Ergebnis.** Steht jetzt in `AGENTS.md`.
- **Zweiter Fund, der die Deutung korrigiert hat:** `steps` misst nur in
  *abgeschlossenen* Runden die Weglänge — die Runde endet mit der letzten Münze.
  Getrennt ausgewertet hat v1 seine 109 abgeschlossenen Runden in **124,1** Schritten
  gelaufen, also schon so schnell wie die Referenz. Die Schrittkosten haben nicht den
  Weg verkürzt, sondern die Zyklen beseitigt; die Navigation war die ganze Zeit in
  Ordnung. Ohne die Abschlussquote danebenzustellen hätte ich das falsch in den
  Bericht geschrieben.
- **Ergebnis v3:** 50,00 Münzen [50,00; 50,00] in 300/300 Runden, **123,8 Schritte**
  gegen 125,3 der Referenz (gepaart −1,45 [−2,48; −0,41], KI schließt die Null aus),
  0,06 ungültige Aktionen. `argmax = coin_dir` in **100 %** der Zustände.
- **Nächster Schritt / Ideen:**
  1. Stufe 2 (`classic`, keine Gegner): Gefahren-Merkmale bauen — „liege ich im
     Explosionsradius?", „habe ich einen Fluchtweg?" — **vor** `BOMB` im Aktionsraum.
     Vorher Unittest für die Explosionsgeometrie (steht seit 31.07. offen).
  2. Zustandsraum wächst deutlich; prüfen, ob die Tabelle noch trägt oder ob hier
     Modell B (DQN) anfangen sollte.
  3. γ und Höhe der Schrittkosten bewusst *nicht* weiter optimiert — die Weglänge liegt
     schon unter der Referenz, das zahlt nicht aufs Turnier ein.
  4. `results/eval/baselines/` als Unterordner: mit Benedict und Ben abstimmen, ob das
     Namensschema aus `AGENTS.md` entsprechend angepasst wird.

### 2026-07-31 — Messkette steht, Konzept festgelegt

- **Stand:** `tools/` ist gebaut und getestet, `KONZEPT.md` liegt vor. Mein Agent
  (`schmaxi_coin_collector`) ist noch unverändert die Vorlage — kein Feature, kein Lernen.
- **Gemacht:**
  - `tools/evaluate.py` — Evaluation mit Per-Runde-Per-Agent-Statistik als CSV + `.meta.json`
    (Commit, Seed, `settings.py`-Abzug). Steuert `BombeRLeWorld` direkt, fasst keine
    Framework-Datei an.
  - `tools/analyze.py` — Bootstrap-KIs, gepaarter Vergleich, Weglass-Ablation, Markdown-Tabellen
    und Abbildungen (`--plot`).
  - `tools/trainlog.py` — eine Zeile pro Episode für Lernkurven, muss noch in `train.py`.
  - `tools/README.md`, `KONZEPT.md`, `AGENTS.md` aktualisiert.
  - Demolauf mit den mitgelieferten Agenten (100 Runden) zum Prüfen der Plots.
- **Entscheidung + Warum:**
  - **Modell A tabellarisches Q-Learning, Modell B DQN.** A ist Absicherung, Vorlesungsbezug
    und schnelle Baseline; B, weil wir GPUs haben. Go/No-Go für B nach Arbeitsstand —
    schlägt es `rule_based` nicht rechtzeitig, geht A ins Turnier.
  - **Messkette vor dem ersten Training.** `--save-stats` reicht nicht: nur Lebenszeit-Summen
    pro Agent und Rundenwerte über alle Agenten summiert, also kein CI für einen einzelnen Agenten.
  - **Fester Seed `20260731` + Reseeding pro Runde.** Der Welt-RNG wird auch in jedem Schritt
    gezogen (Agentenreihenfolge), deshalb driften Spielfelder ab Runde 2 auseinander. Nachgemessen
    per Arena-Hash.
  - **Statistische Ehrlichkeit:** CI enthält die Null → nicht gezeigt, kommt trotzdem in den Bericht.
- **Offen / Nächster Schritt:**
  1. Arbeitsteilung in KONZEPT.md §7 mit Benedict und Ben ausfüllen.
  2. `results/` steht in `.gitignore` — Messdaten wären unversioniert. Im Team klären (§6.7).
  3. Merkmalssatz Stufe 1 bauen (`coin_dir` + `neighbours`, ≈405 Zustände), vorher Unittest
     für die Explosionsgeometrie.
  4. Baselines messen (`random_agent`, `rule_based_agent` auf `coin-heaven`), bevor trainiert wird.
- **Notiz:** Pairing bringt bei sehr verschiedenen Agenten fast nichts (gemessen r ≈ 0,24 → 1,16×
  schmaler). Nutzen steigt mit der Ähnlichkeit der Varianten; der eigentliche Gewinn ist
  Reproduzierbarkeit. Die Formulierung „um ein Vielfaches schmaler" in KONZEPT.md §6.2 und
  `evaluate.py` ist zu stark und muss noch korrigiert werden.

<!-- Neueste Einträge oben. Format:
### JJJJ-MM-TT — Kurztitel
- **Stand:** wo ich aufgehört habe
- **Gemacht:** was passiert ist
- **Entscheidung + Warum:** ...
- **Nächster Schritt / Ideen:** ...
-->
