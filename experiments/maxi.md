# Versuchsprotokoll — Maxi

Ergebnisbuch, ein Eintrag pro Experiment. Getrennt von `MAXI.md`: dort steht, *warum*
ich etwas entschieden habe, hier stehen die *Zahlen*. Die Einträge werden im Bericht
fast wörtlich zu Tabellenzeilen.

Regeln, an die ich mich halte:

- **Vorhersage vor der Messung.** Steht sie nicht vorher da, war es kein Experiment,
  sondern eine Beobachtung — dann schreibe ich das auch so hin.
- **Eine Änderung pro Eintrag.** Wo ich das verletzt habe, laufen Kontrollläufe mit,
  die die Änderungen wieder trennen.
- **Commit-Hash mitschreiben** (steht in `results/*/<label>.meta.json`).
- **Negative Ergebnisse bleiben stehen.**
- Fester Seed `20260731`, 300 Runden für jede berichtete Zahl.

Urteil: **BESSER** · **SCHLECHTER** · **nicht gezeigt** (KI enthält die Null).

---

## E00 — Referenzmessung: mitgelieferte Agenten auf `coin-heaven`

**Kein Experiment, sondern eine Referenzmessung.** Es gab nichts vorherzusagen und keine
Änderung zu bewerten; der Eintrag legt die Skala fest, auf der alle Stufe-1-Einträge
gelesen werden.

- **Frage:** Wo liegen Boden und Decke auf Stufe 1, bevor ich anfange zu trainieren?
- **Aufbau:** jeder Agent **allein** (`--opponents none`), `coin-heaven`, 300 Runden,
  Seed `20260731`, Commit `0c33107`.
- **Daten:** `results/eval/baselines/baseline_*__task1.csv`

```bash
uv run python tools/evaluate.py --agents <agent> --opponents none \
    --scenario coin-heaven --n-rounds 300 --seed 20260731 --label baseline_<agent>__task1
```

| Agent | `coins` | 95-%-KI | `steps` | 95-%-KI | `invalid` |
|---|---|---|---|---|---|
| `random_agent` | 1,77 | [1,55; 2,00] | 22,4 | [20,4; 24,6] | 9,56 |
| `peaceful_agent` | 18,70 | [18,13; 19,27] | 400,0 | [400,0; 400,0] | 148,10 |
| `coin_collector_agent` | **50,00** | [50,00; 50,00] | 125,3 | [124,3; 126,3] | 0,00 |
| `rule_based_agent` | **50,00** | [50,00; 50,00] | 124,8 | [123,8; 125,8] | 0,00 |

**Konsequenzen für alles Weitere:**

1. `coins` ist auf Stufe 1 **kein Optimierungsziel, sondern ein Bestehenskriterium.**
   Beide Referenzagenten holen in *jeder* der 300 Runden alle 50 Münzen, Varianz null.
   Sobald mein Agent bei 50 liegt, unterscheidet `coins` nichts mehr; die eigentliche
   Zielgröße ist dann **`steps`** mit Marke **≈ 125**.
2. `coin_collector_agent` und `rule_based_agent` sind hier nicht unterscheidbar
   (125,3 vs. 124,8, überlappende KIs) — ohne Kisten läuft die Bombenlogik nie an.
   Für Stufe 1 ist `coin_collector_agent` die **einzige** Referenz.
3. Die 125 Schritte sind schlagbar: die Referenz läuft greedy zur *nächsten* Münze,
   das ist nicht die optimale Rundreise.
4. `peaceful_agent` ist der aussagekräftigere Boden: stirbt nie, sammelt aber nur 18,7
   Münzen bei **148 ungültigen Aktionen** — das Profil „bewegt sich, navigiert nicht".

---

## E01 — Aktionsmaske + schnellerer ε-Zerfall (Stufe 1)

- **Agent:** `maxi_coin_collector`, tabellarisches Q-Learning
  - Merkmale (`features.py`): `(coin_dir, up, right, down, left)` — BFS-Richtung zur
    nächsten Münze + 4 Blockiert-Bits. 5 × 16 = 80 mögliche Zustände.
  - α = 0,1 · γ = 0,9 · ε: 1,0 → 0,05
  - Rewards: `COIN_COLLECTED` +5 · `INVALID_ACTION` −1 · `WAITED` −0,5 ·
    **Schrittkosten deaktiviert** (`#reward_sum -= 0.1`)
- **Training:** je 1000 Runden `coin-heaven`, Commits `8c70785` / `157c181`
- **Messung:** 300 Runden, Seed `20260731`, ε = 0

### Vorbedingung: drei Fehler, die den Lauf überhaupt erst ermöglicht haben

Der erste Trainingsversuch ist sofort abgestürzt. Gehört nicht zum Experiment, muss aber
protokolliert sein, weil sich der Commit sonst nicht erklärt:

1. `callbacks.py` — `self.model` war noch der Wahrscheinlichkeitsvektor aus der Vorlage
   (`np.array`, Länge 6), nicht die Q-Tabelle. `features not in self.model` hat damit
   elementweise ein 5-Tupel gegen ein 6-Array verglichen → `ValueError`.
2. `train.py` — `numpy` war nie importiert.
3. `update_q_table` — kein Schutz gegen `old_game_state`/`action` = `None`.

### Ausgangslage (v1, unverändert)

Alle 1000 Episoden endeten mit `KILLED_SELF`, `SURVIVED_ROUND` = 0,00 durchgehend.
Der Agent lebte im Schnitt 18 von 400 Schritten. Ursache: `BOMB` ist im Aktionsraum,
aber **kein Merkmal sieht eine Bombe** — Weglaufen ist nicht lernbar, jede per ε
gezogene Bombe ist tödlich. Auf `coin-heaven` hat eine Bombe zudem keinerlei Nutzen.

### Änderungen

- **A — Aktionsmaske:** `ALLOWED_ACTIONS = [UP, RIGHT, DOWN, LEFT]`. `BOMB` und `WAIT`
  kommen erst auf Stufe 2 zurück, zusammen mit den Gefahren-Merkmalen. Die Q-Vektoren
  bleiben 6 lang, das `max` im Bellmann-Update läuft nur über die erlaubten Indizes.
- **B — ε-Zerfall 0,9995 → 0,997.** 0,9995^1000 = 0,61: der alte Lauf hat die
  Politik nie bei kleinem ε ausgeführt. 0,997^1000 ≈ 0,05 erreicht `EPSILON_END`
  innerhalb des Laufs.

Weil das zwei Änderungen auf einmal sind, laufen zwei Kontrollen mit (je 1000 Runden):
**v1a** nur Maske, **v1b** nur Zerfall.

### Ergebnis Training (`results/train/`, Abb. `results/figures/maxi_task1_train_*.png`)

Mittelwerte der ersten bzw. letzten 100 Episoden:

| Lauf | | `score` | `steps` | `KILLED_SELF` | `invalid` | ε |
|---|---|---|---|---|---|---|
| v1 original | erste | 1,02 | 10,9 | 1,00 | 4,2 | 0,98 |
| | letzte | 3,68 | 18,3 | **1,00** | 4,0 | 0,62 |
| v1a nur Maske | erste | 19,79 | 400 | 0,00 | 143,6 | 0,98 |
| | letzte | 47,64 | 373,7 | 0,00 | 78,2 | 0,62 |
| v1b nur Zerfall | erste | 1,59 | 13,6 | 1,00 | 4,2 | 0,86 |
| | letzte | 27,61 | 79,4 | **0,62** | 2,6 | 0,06 |
| v2 beides | erste | 31,50 | 400 | 0,00 | 123,1 | 0,86 |
| | letzte | **48,16** | 167,0 | 0,00 | 4,8 | 0,06 |

Die Kontrollen trennen die Wirkung sauber, und keine der beiden Änderungen genügt allein:

- **Die Maske behebt das Sterben** (Selbstmorde sofort 1,00 → 0,00), lässt den Agenten
  wegen ε = 0,62 am Ende aber weiter herumfuchteln: 78 ungültige Aktionen pro Runde.
- **Der Zerfall allein** bringt die Politik überhaupt erst zum Einsatz (`invalid`
  4,2 → 2,6), aber 62 % der Episoden enden weiter im Selbstmord — mit `BOMB` im
  Aktionsraum und ohne Gefahren-Merkmal ist das nicht wegtrainierbar.
- Zusammen: 48,2 Münzen, 4,8 ungültige Aktionen, null Selbstmorde.

### Ergebnis Messung mit ε = 0 — und hier kippt es

| Lauf | `coins` | 95-%-KI | `steps` | `invalid` | `suicides` |
|---|---|---|---|---|---|
| `maxi_q_v1__task1` | **32,74** | [30,75; 34,76] | 293,2 | 0,16 | 0,02 |
| `maxi_q_v2__task1` | **1,45** | [1,29; 1,62] | 400,0 | 157,76 | 0,00 |
| Referenz `coin_collector_agent` | 50,00 | [50,00; 50,00] | 125,3 | 0,00 | — |

Gepaart (v1 → v2, identische Arenen, `maxi_task1_v1_vs_v2.png`):
`coins` **−31,29 [−33,35; −29,29]**, `invalid` **+157,60 [+135,12; +180,04]`.

**Urteil: SCHLECHTER.** Der Lauf, der im Training mit Abstand am besten aussah, ist der
mit Abstand schlechteste, sobald die Tabelle eingefroren ist. Das ist das eigentliche
Ergebnis dieses Eintrags.

*(v1 musste für diese Zeile nachtrainiert werden — der ursprüngliche v1-Lauf hatte die
Modelldatei bereits überschrieben. Reproduktion als `q_v1_repro`, Commit `157c181`.
Die v2-Messung wurde einmal wiederholt und war auf drei Nachkommastellen identisch,
es ist also kein Artefakt einer veralteten Modelldatei.)*

### Ursache

Zwei Befunde aus der gespeicherten Q-Tabelle:

| | Zustände | mittlerer Q-Pegel | mittlere Spreizung über die 4 Züge | argmax = `coin_dir` |
|---|---|---|---|---|
| v1 | 32 | 8,39 | 5,78 | **84 %** |
| v2 | 39 | 9,07 | 3,05 | **59 %** |

1. **Ohne Schrittkosten flacht die Tabelle ab.** Münze +5, Laufen gratis: jeder Weg
   sammelt am Ende jede Münze, also haben alle vier Züge fast denselben Rückfluss.
   Die Differenz, die „geh Richtung Münze" kodiert, wird vom gemeinsamen Pegel
   überdeckt. v2 trainiert länger und in 400-Schritt-Episoden, sättigt dadurch
   *stärker* — und verliert genau deshalb den `coin_dir`-Bezug (84 % → 59 %).
   **Mehr Training hat die eingefrorene Politik schlechter gemacht.**
2. **Eine deterministische, ortsblinde Politik hat absorbierende Zyklen.** Die
   `invalid`-Verteilung von v2 ist nicht verrauscht, sondern zweigipflig: **119 von 300
   Runden haben 399 ungültige Aktionen** (der Agent verklemmt sich in Schritt 1 gegen
   eine Wand und wiederholt dieselbe ungültige Aktion 399-mal — die Position ändert
   sich nicht, also ändern sich die Merkmale nicht, also ändert sich die Aktion nicht),
   **156 Runden haben exakt 0** und pendeln stattdessen zwischen zwei Feldern.
   Im Training bricht ε = 0,05 diese Zyklen ständig auf, und die Tabelle wird
   fortlaufend weiter verändert — deshalb sieht das Trainingslog gesund aus.
   Das ist genau der Effekt, den Benedict für seine Wandbits-Baseline vorhergesagt hat.

### Was ich daraus mache

- **Trainingskurven sind kein Ersatz für eine Messung mit ε = 0.** Ab jetzt gehört zu
  jedem Trainingslauf eine 300-Runden-Messung, bevor ein Ergebnis behauptet wird.
- Maske und Zerfall bleiben (der Selbstmord ist real behoben und in v1 nur deshalb
  nicht sichtbar, weil dort ohnehin nichts funktioniert hat).
- Nächster Schritt ist **eine** Änderung: Schrittkosten aktivieren.

---

## E02 — Schrittkosten −0,1 pro Schritt (geplant, Vorhersage vor dem Lauf)

- **Frage:** Stellt eine Schrittkosten-Komponente die Spreizung der Q-Werte wieder her,
  so dass die eingefrorene Politik der Münzrichtung folgt statt in Zyklen zu laufen?
- **Änderung ggü. E01-v2:** `reward_from_events` — `reward_sum -= 0.1` aktivieren.
  Sonst nichts.

### Vorhersage

1. `argmax = coin_dir` steigt deutlich über die 59 % von v2, Zielbereich > 90 %.
   Das ist die eigentliche Prüfgröße; alles andere folgt daraus.
2. `invalid` fällt von 157,8 auf < 5. Insbesondere verschwindet der Gipfel bei 399:
   ein Zug gegen die Wand kostet dann −1 **und** −0,1, während jeder gültige Zug
   Fortschritt in Richtung +5 bringt.
3. `coins` > 45. Ich erwarte **nicht** sofort 50 — 7 der 80 Zustände waren in v2 nie
   besucht und stehen weiter auf null.
4. `steps` bleibt zunächst deutlich über der Referenzmarke 125, weil γ = 0,9 bei
   −0,1 pro Schritt nur schwach auf Kürze drückt. Die Weglänge ist ein eigenes,
   späteres Experiment (γ und die Höhe der Schrittkosten).
5. Fällt `coins` **nicht**, liegt der Fehler nicht in der Belohnung, sondern im
   Merkmalssatz — dann ist der nächste Verdacht, dass `coin_dir` allein den Zustand
   zu grob beschreibt (Sackgassen sind nicht unterscheidbar).

### Ergebnis

*(nach der Messung ausfüllen)*
