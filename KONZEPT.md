# Konzept & Vorgehen

Gemeinsame Arbeitsgrundlage für Benedict, Ben und Maxi. `AGENTS.md` ist die knappe
technische Spec — dieses Dokument erklärt das *Warum* und legt fest, wie wir messen.

Stand: 31.07.2026 · Deadlines: Code **21.09.**, Bericht **28.09.**

---

## 1. Worauf es bei der Note ankommt

Die Aufgabenstellung ist an dieser Stelle ungewöhnlich deutlich: Turnierplatzierung
ist ein Faktor, aber *"the quality of your report and a systematic (scientific)
approach to agent design, optimization, and testing will carry much more weight"*.
Und weiter, zu *Experiments and Results*: **"This is the most important section of
the report."**

Daraus folgt unsere wichtigste Arbeitsregel:

> **Keine Änderung am Agenten ohne Messung davor und danach.**

Ein Agent, der 6,0 Punkte erreicht und bei dem wir für zwölf Designentscheidungen
sauber belegen können, ob sie geholfen haben, ist für die Note besser als einer mit
8,0 Punkten, der durch Herumprobieren entstanden ist. Deshalb steht die
Messinfrastruktur (Abschnitt 6) *vor* dem ersten Trainingslauf — sie ist bereits
gebaut und liegt in `tools/`.

Der zweite Grund ist praktisch: Ohne Messung wisst ihr schlicht nicht, ob eine
Änderung geholfen hat. Die Rundenvarianz in diesem Spiel ist so hoch, dass ein
Unterschied von 0,5 Punkten über 20 Runden fast immer Zufall ist.

---

## 2. Die zwei Modelle

Die Aufgabe verlangt mindestens zwei Modelle, mindestens eines mit Techniken aus
der Vorlesung. Wir bauen:

### Modell A — Tabellarisches Q-Learning auf handgebauten Merkmalen

Unsere Absicherung und der Vorlesungsbezug. Wir komprimieren `game_state` auf
wenige diskrete Merkmale und lernen eine Q-Tabelle darüber.

*Warum zuerst:* Es konvergiert in Minuten statt Tagen, es ist vollständig
interpretierbar (wir können in die Tabelle schauen und sehen, was der Agent gelernt
hat), die 0,5-Sekunden-Grenze ist nie ein Thema, und es liefert die Baseline, gegen
die alles andere gemessen wird. Die Aufgabenstellung weist ausdrücklich darauf hin,
dass in der Vergangenheit einfache, gut getunte Modelle gewonnen haben.

*Grenze:* Unsere Leistung ist gedeckelt durch die Qualität der Merkmale, und
Zustands-Aliasing (zwei objektiv verschiedene Lagen landen in derselben Tabellenzelle)
wird uns beschäftigen.

### Modell B — Deep Q-Network auf dem rohen Brett

Wir haben Zugriff auf starke GPUs — das ist ein echter Vorteil gegenüber anderen
Gruppen, und wir wollen es einmal gemacht haben. Das Netz lernt die Merkmale selbst
statt sie von uns vorgegeben zu bekommen.

*Risiko, das wir ernst nehmen müssen:* Die Aufgabenstellung warnt explizit, dass
Deep-RL-Ansätze in früheren Jahren wiederholt zum Turnier nicht konvergiert waren.
Unser Gegenmittel: früh anfangen, Warmstart durch Verhaltensklonen (Abschnitt 5),
Symmetrie-Augmentierung für den Faktor 8 in der Stichprobeneffizienz, und ein
hartes Abbruchkriterium — **wenn Modell B am 07.09. den `rule_based_agent` nicht
schlägt, geht Modell A ins Turnier.** Modell B bleibt trotzdem im Bericht; ein
dokumentierter Fehlschlag mit sauberer Analyse ist ein vollwertiges Ergebnis.

*Inferenz auf der CPU:* Ein kleines CNN auf 17×17 braucht wenige Millisekunden pro
Zug. Das ist unkritisch — aber `tools/evaluate.py` misst `think_max_ms` mit, damit
wir es nicht auf gut Glück annehmen.

### Was beide teilen

Merkmalsextraktion (die BFS-Hilfsfunktionen), Belohnungsdesign,
Evaluationsprotokoll und Trainings-Logging sind **gemeinsam**. Nur so sind die
Modelle vergleichbar — und nur so erfüllen wir die Vorgabe, dass nicht jeder sein
eigenes Modell in Isolation baut.

---

## 3. Merkmalsentwurf

### Die harte Regel

> *"you are not allowed to submit a model which does not learn from the features
> you define. This would disallow, for example, a feature that deterministically
> returns the action which results in the best move."*

Erlaubt: „Richtung zur nächsten Münze", „Anzahl Kisten im Explosionsradius",
„bin ich in Gefahr". Verboten: „beste Aktion". Die Trennlinie ist, ob das Merkmal
eine *Eigenschaft des Zustands* beschreibt oder bereits die Entscheidung trifft.

Im Zweifel: Ein Merkmal darf nicht so heißen, dass man es direkt als Aktion
zurückgeben könnte. `bomb_is_a_good_idea` wäre grenzwertig; die Zerlegung in
`crates_in_blast`, `opponent_in_blast` und `escape_exists_after_bomb` ist sauber,
weil das Modell erst lernen muss, wie es die drei kombiniert.

### Merkmalssätze wachsen mit der Task-Leiter

Das ist bewusst so gestaffelt: kleiner Zustandsraum zuerst, schnelle Konvergenz,
schnelles Feedback. Jede Erweiterung ist ein eigenes Experiment mit Vorher/Nachher-Messung.

**Stufe 1 — `coin-heaven`, Navigation** (≈ 405 Zustände)

| Merkmal | Werte | Beschreibung |
|---|---|---|
| `coin_dir` | 5 | BFS-Richtung zur nächsten erreichbaren Münze: `none/up/right/down/left` |
| `neighbours` | 81 | pro Nachbarfeld `{frei, blockiert, tödlich}` = 3⁴ |

3⁴ × 5 = 405 Zustände × 6 Aktionen = 2 430 Q-Werte. Das konvergiert in wenigen
tausend Episoden und ist unser Funktionstest für die gesamte Pipeline.

**Stufe 2 — `classic` ohne Gegner, Bomben und Flucht** (≈ 260 000)

| Merkmal | Werte | Beschreibung |
|---|---|---|
| `target_dir` | 5 | Richtung zum nächsten Ziel (Münze, sonst Kiste) |
| `neighbours` | 81 | wie oben |
| `danger` | 4 | `{sicher, t≥3, t=2, t≤1}` — Explosionszeit auf dem eigenen Feld |
| `escape_dir` | 5 | BFS-Richtung zum nächsten sicheren Feld (nur relevant bei `danger>0`) |
| `bomb_available` | 2 | aus `game_state['self'][2]` |
| `crates_in_blast` | 4 | `{0,1,2,3+}` Kisten, die eine Bombe hier zerstören würde |
| `escape_after_bomb` | 2 | existiert ein Fluchtweg, wenn ich jetzt lege |

**Stufe 3/4 — Gegner** (× 15)

| Merkmal | Werte | Beschreibung |
|---|---|---|
| `opponent_dir` | 5 | Richtung zum nächsten Gegner |
| `opponent_dist` | 3 | `{angrenzend, nah ≤3, fern}` |

Die Q-Tabelle wird als `dict` geführt, nicht als dichtes Array — nur besuchte
Zustände belegen Speicher, und das sind sehr viel weniger als das Produkt der
Kardinalitäten.

### Aliasing im Blick behalten

Führt eine Besuchszählung pro Zustand mit. Zwei Warnsignale: Zustände, die sehr oft
besucht werden, aber deren Q-Werte nicht konvergieren (dort verstecken sich mehrere
echte Situationen), und Zustände mit sehr wenigen Besuchen (dort ist der Q-Wert
Rauschen). Beides ist Material für den Bericht.

### Bombenflucht ist der kritische Teil

Die Aufgabenstellung sagt es direkt: *"Escaping bombs is a crucial capability for
good tournament performance, so place proper emphasis on this step."* Die meisten
Agenten sterben an der eigenen Bombe. `escape_dir` und `escape_after_bomb` brauchen
eine korrekte BFS, die Explosionsgeometrie berücksichtigt: Reichweite 3, wird von
Steinwänden geblockt, geht **nicht** um Ecken, und die Explosion bleibt
`EXPLOSION_TIMER = 2` Schritte liegen. Diese Funktion sollte einen eigenen Unittest
bekommen — ein Fehler dort vergiftet alles Weitere.

---

## 4. Belohnungsdesign

Die Basisbelohnung (Münze +1, Kill +5) ist zu dünn: Ein zufälliger Agent sieht sie
praktisch nie. Wir brauchen ein dichtes Signal — aber vorsichtig, denn die
Hilfsbelohnungen existieren im Turnier nicht.

### Potenzialbasiertes Shaping

Die Aufgabenstellung verweist in einer Fußnote auf Ng et al. (1999). Die Aussage:
Eine Zusatzbelohnung der Form

```
F(s, s') = γ · Φ(s') − Φ(s)
```

verändert die optimale Strategie **beweisbar nicht**. Φ ist eine Funktion des
*Zustands*, etwa die negative Distanz zur nächsten Münze.

Das ist genau das Gegenmittel gegen die Falle, vor der die Aufgabenstellung warnt:
Belohnung fürs Hinlaufen ohne gleich große Strafe fürs Weglaufen erzeugt einen
Agenten, der zwischen zwei Feldern pendelt und Belohnung farmt statt Fortschritt zu
machen. Bei potenzialbasiertem Shaping heben sich Hin- und Rückweg exakt auf.

Das ist ein geschenkter Absatz für das Kapitel *Methods* — mit Literaturreferenz,
die die Aufgabenstellung selbst nahelegt.

### Vorschlag für den Start

| Ereignis | Wert | Begründung |
|---|---|---|
| `COIN_COLLECTED` | +5 | Hauptziel Stufe 1 |
| `KILLED_OPPONENT` | +25 | im Spiel 5× so viel wert wie eine Münze |
| `KILLED_SELF` | −30 | teurer als ein Kill wert ist |
| `GOT_KILLED` | −15 | |
| `CRATE_DESTROYED` | +1 | Zwischenziel, bewusst klein |
| `COIN_FOUND` | +2 | |
| `INVALID_ACTION` | −3 | verschwendet einen Zug |
| `WAITED` | −0,5 | verhindert Erstarren |
| Schrittstrafe | −0,2 | Druck zur Effizienz |
| Φ-Shaping Distanz | γΦ' − Φ | Φ = −BFS-Distanz zum Ziel |

Eigene Ereignisse, die sich lohnen: `MOVED_OUT_OF_BLAST`, `MOVED_INTO_BLAST`,
`BOMB_WITH_NO_ESCAPE`, `USELESS_BOMB` (Bombe ohne Kiste und ohne Gegner in
Reichweite).

**Diese Zahlen sind Startwerte, keine Wahrheit.** Sie sind Hyperparameter und
gehören in die systematische Optimierung — die Aufgabenstellung nennt
Hyperparameteroptimierung explizit als Bewertungspunkt.

---

## 5. Vier Techniken, die den Unterschied machen

**Symmetrie-Augmentierung.** Das Brett hat die volle Symmetriegruppe des Quadrats:
vier Drehungen mal Spiegelung, acht Elemente. Jeder Übergang lässt sich achtfach
verwerten, wenn die Aktion mitgedreht wird. Für Modell B ist das der Unterschied
zwischen konvergiert und nicht konvergiert; für Modell A eine Alternative dazu:
Zustände kanonisieren (immer in die lexikographisch kleinste der acht Varianten
drehen) und damit die Tabelle um bis zu Faktor 8 verkleinern. Beide Varianten sind
ein sauberes Experiment wert.

*Achtung beim Testen:* Die Startecken sind zufällig, aber das Wandmuster ist
symmetrisch — prüft mit einem Unittest, dass Drehung von Zustand *und* Aktion
zusammenpassen. Ein Vorzeichenfehler hier ist praktisch unauffindbar.

**Warmstart durch Verhaltensklonen.** Zufälliges ε-greedy entdeckt die Sequenz
„Bombe legen → drei Felder fliehen → warten → zurück" praktisch nie; die
Wahrscheinlichkeit ist zu klein. Die Aufgabenstellung erlaubt ausdrücklich, den
`rule_based_agent` Trainingsdaten erzeugen zu lassen. Also: erst überwacht auf
dessen Zügen vorlernen, dann mit RL feinschleifen. Das spart Wochen. Wichtig fürs
Protokoll: Der abgegebene Agent muss *gelernt* sein, nicht regelbasiert — Klonen
als Initialisierung ist Lernen, das Kopieren der Regeln wäre es nicht.

**Curriculum entlang der Task-Leiter.** Stufe 1 → 2 → 3 → 4, mit einem Vorbehalt:
katastrophales Vergessen. Wenn Stufe 2 nur noch auf `classic` trainiert, verlernt
der Agent die effiziente Navigation aus Stufe 1. Gegenmittel: Szenarien mischen
(z. B. 20 % der Episoden aus der vorherigen Stufe) und **nach jeder Stufe auch die
vorherige neu messen**.

**Self-Play für Stufe 4.** Mehrere Instanzen des eigenen Agenten gegeneinander,
mit erhöhtem ε gegen zu frühe Konvergenz. Praktisch: einen Pool früherer
Versionen als Gegner ziehen, damit der Agent nicht gegen eine einzige Strategie
überanpasst.

---

## 6. Messinfrastruktur

Liegt in `tools/` und ist einsatzbereit. Der Rest dieses Abschnitts ist die
Vereinbarung, wie wir sie benutzen — sie ist nur nützlich, wenn wir drei dieselben
Zahlen produzieren.

### 6.1 Warum nicht `main.py --save-stats`

Das Framework schreibt zwei Dinge: Lebenszeit-Summen pro Agent, und Rundenwerte
**über alle Agenten summiert**. Beides erlaubt kein Konfidenzintervall für einen
einzelnen Agenten, weil wir dessen Score in einer *einzelnen* Runde nie sehen.
Deshalb `tools/evaluate.py`, das `BombeRLeWorld` direkt steuert und die
Per-Runde-Per-Agent-Statistik abgreift, bevor die nächste Runde sie zurücksetzt.
Framework-Dateien bleiben unangetastet — das Skript funktioniert also auch, wenn
das Framework fürs Turnier zurückgesetzt wird.

### 6.2 Arena-Matching — der eigentliche Trick

`main.py --seed N` initialisiert den Welt-RNG **einmal**. Aus demselben RNG wird
aber in jedem Schritt gezogen (die Aktivierungsreihenfolge der Agenten). Wie viele
Ziehungen eine Runde verbraucht, hängt also davon ab, wie lange sie gedauert hat —
und damit unterscheiden sich ab Runde 2 die Spielfelder zwischen zwei Agenten,
selbst bei identischem `--seed`.

`evaluate.py` setzt den RNG vor jeder Runde neu auf `base_seed + round_index`.
Runde *i* hat dann für jeden je gemessenen Agenten exakt dieselben Kisten, Münzen
und Startecken. Empirisch geprüft:

```
mit Reseeding : ['0b6695e2', 'ff1dd92e', '83b17044', ...]   ← zwei verschiedene
                ['0b6695e2', 'ff1dd92e', '83b17044', ...]      Aufstellungen, identisch
ohne Reseeding: ['0b6695e2', '5e367900', 'afe98052', ...]   ← driften ab Runde 2
                ['0b6695e2', '759a805a', '76e1c5b0', ...]      auseinander
```

Das macht Vergleiche **gepaart**: Wir messen nicht zwei verrauschte Mittelwerte
gegeneinander, sondern die Differenz Runde für Runde. Das kürzt „diese Arena war
großzügig" heraus und lässt den Effekt der Änderung übrig. Im Test war das
Konfidenzintervall auf der gepaarten Differenz bei nur 20 Runden bereits ±0,15
Punkte — ungepaart bräuchte man dafür ein Vielfaches an Runden.

**Konsequenz für uns: `--seed` niemals ändern.** Standard ist `20260731`. Wer den
Seed variiert, kann seine Zahlen nicht mit unseren vergleichen.

### 6.3 Metriken

`evaluate.py` schreibt pro Runde und Agent eine Zeile:

| Spalte | Bedeutung |
|---|---|
| `score` | **Primärmetrik** — Punkte der Runde (Münze 1, Kill 5) |
| `coins`, `kills`, `suicides`, `crates`, `bombs` | Ereigniszähler |
| `survived` | 1, wenn der Agent die Runde überlebt hat |
| `steps` | Schritte, die der Agent gelebt hat |
| `moves`, `invalid` | Bewegungen bzw. ungültige Aktionen |
| `think_mean_ms`, `think_max_ms`, `think_over_limit` | Rechenzeit gegen das 0,5-s-Limit |
| `round`, `seed`, `slot`, `agent`, `code` | Zuordnung |

Die Diagnosemetriken sind wichtiger, als sie aussehen. `suicides` ist der ehrlichste
Fortschrittsindikator für Stufe 2; `invalid` verrät, ob der Agent gegen Wände läuft;
`think_max_ms` ist unsere einzige Absicherung gegen einen Timeout im Turnier.

Daneben schreibt `evaluate.py` eine `.meta.json` mit Git-Commit, Seed, Szenario und
einem **Abzug der `settings.py`-Werte**. Letzteres, weil „ich kann deine Zahl nicht
reproduzieren" fast immer daran liegt, dass jemand fürs Training die Settings
verstellt hatte.

### 6.4 Protokoll

| Zweck | Runden | Gegner | Wann |
|---|---|---|---|
| Schnellcheck während der Entwicklung | 100 | passend zur Stufe | jederzeit |
| **Standardmessung** | **300** | passend zur Stufe | für jede berichtete Zahl |
| Abschlussmessung | 1000 | `rule_based` | vor Abgabe, für den Bericht |
| Sanity-Check | 100 | `random` | nach größeren Umbauten |

```bash
# Stufe 1
uv run python tools/evaluate.py --agents schmaxi_coin_collector \
    --scenario coin-heaven --n-rounds 300 --label maxi_q_v1__task1

# Stufe 4 (Turnierbedingungen)
uv run python tools/evaluate.py --agents schmaxi_coin_collector \
    --opponents rule_based --n-rounds 300 --label maxi_q_v1__task4

# Auswerten
uv run python tools/analyze.py results/eval/maxi_q_v1__task4.csv
uv run python tools/analyze.py --compare results/eval/maxi_q_v1__task4.csv \
                                         results/eval/maxi_q_v2__task4.csv
```

### 6.4a Abbildungen für den Bericht

`--plot` schreibt nach `results/figures/`. Braucht matplotlib: `uv add matplotlib`
(nur für die Analyse — **nicht** in die abgegebene `requirements.txt`).

```bash
# Balkendiagramm mit Fehlerbalken, ein Feld pro Metrik
uv run python tools/analyze.py results/eval/maxi_q_v3__task4.csv --plot

# Forest-Plot: eine Version gegen eine andere, alle Metriken
uv run python tools/analyze.py --compare results/eval/maxi_q_v2__task4.csv \
                                         results/eval/maxi_q_v3__task4.csv --plot

# Ablationsstudie: viele Varianten gegen eine Baseline, eine Metrik
uv run python tools/analyze.py --ablation results/eval/q_base__task2.csv \
       results/eval/q_ohne_shaping__task2.csv \
       results/eval/q_ohne_symmetrie__task2.csv \
       results/eval/q_kleine_features__task2.csv \
       --metric score --plot
```

Der **Forest-Plot der Ablation** ist die wichtigste Abbildung für das
*Experiments*-Kapitel: eine Zeile pro Designentscheidung, Punktschätzer mit
Konfidenzintervall, gestrichelte Null-Linie. Grün heißt besser, rot schlechter,
grau heißt „Intervall kreuzt die Null, nicht gezeigt". Der Leser sieht auf einen
Blick, welche unserer Entscheidungen durch Daten gedeckt sind — und genau das
verlangt die Aufgabenstellung.

`results/figures/` enthält nur generierte PNGs; die lassen sich jederzeit aus den
CSVs neu erzeugen und müssen nicht versioniert werden.

Dateinamen: `<person>_<modell>_<version>__<stufe>.csv`, z. B.
`maxi_q_v3__task2.csv`, `benedict_dqn_v1__task4.csv`.

### 6.5 Wann eine Änderung als Verbesserung gilt

`analyze.py --compare` gibt für jede Metrik die gepaarte Differenz mit
95-%-Bootstrap-Konfidenzintervall aus.

- **Enthält das KI die Null, ist die Verbesserung nicht gezeigt.** Nicht „geht in
  die richtige Richtung" — nicht gezeigt.
- Wir übernehmen eine Änderung, wenn die Primärmetrik signifikant besser ist, oder
  wenn sie neutral ist und eine Diagnosemetrik (v. a. `suicides`) signifikant besser
  wird.
- Ein negatives Ergebnis wird **nicht gelöscht.** Es kommt mit in den Bericht. Ein
  dokumentierter Fehlschlag ist wissenschaftliche Arbeit; ein stillschweigend
  verworfener Versuch ist verlorene Note.

Bootstrap statt Normalapproximation, weil der Rundenscore stark schief verteilt ist
(viele Runden nahe null, wenige große bei Kills) — das Resampling braucht keine
Verteilungsannahme.

### 6.6 Lernkurven

`tools/trainlog.py` schreibt eine Zeile pro Episode: Score, Schritte, geshapte
Belohnung, ε, Ereigniszähler, plus frei definierbare Spalten (TD-Fehler, Loss).
Einbinden in `train.py`:

```python
try:
    from tools.trainlog import TrainLogger
except ImportError:          # tools/ ist nicht Teil der Abgabe
    TrainLogger = None
```

Plot: `uv run python tools/trainlog.py results/train/*.csv --metric score`

**Mehrere Seeds pro Lernkurve.** Eine einzelne Kurve sagt wenig — RL-Training ist
zwischen Läufen erheblich verrauscht. Drei bis fünf Läufe pro Konfiguration, im
Bericht als Band dargestellt.

### 6.7 Offener Punkt: `results/` ist in `.gitignore`

Damit landen unsere Messdaten nicht im Repo. Die CSVs sind klein (300 Runden ≈
50 kB) und sind die Belege für das wichtigste Kapitel des Berichts — sie sollten
versioniert werden. Vorschlag: `results/` freigeben und stattdessen gezielt
`results/**/*.png` und große Binärdateien ausschließen. **Bitte im Team abstimmen,
bevor jemand die `.gitignore` anfasst.**

---

## 7. Arbeitsteilung

Die Aufgabenstellung ist hier ausdrücklich: *"do not split labor such that every
team member works on their separate model — we attach great importance to real
teamwork!"*

Wir teilen deshalb **nach Komponenten**, nicht nach Modellen. Beide Modelle nutzen
dieselbe Merkmalsextraktion, dasselbe Belohnungsschema und dieselbe Messkette.

| Bereich | Hauptverantwortung | Beteiligt |
|---|---|---|
| Merkmalsextraktion, BFS-Bibliothek, Symmetrien | *offen* | alle nutzen sie |
| Q-Learning-Kern, Trainingsschleife, Hyperparameter | *offen* | |
| DQN: Netz, Replay-Buffer, GPU-Training | *offen* | |
| Belohnungsdesign, eigene Ereignisse, Ablationen | *offen* | |
| Messkette, Experimentplanung, Auswertung | *offen* | |

Hauptverantwortung heißt: treibt es voran und kennt den Stand — nicht: arbeitet
allein daran. Für den Bericht müssen ohnehin die Autoren pro Abschnitt markiert
werden, also führt jede/r mit, was er oder sie beigetragen hat. Die Logbücher
(`MAXI.md`, `BENEDICT.md`, `BEN.md`) sind dafür da.

**Bitte in der nächsten Besprechung ausfüllen.**

---

## 8. Meilensteine

Rund sieben Wochen bis zur Code-Abgabe.

| Woche | Ziel | Fertig, wenn |
|---|---|---|
| **KW 32** (bis 09.08.) | Stufe 1 gelöst | Q-Agent sammelt in `coin-heaven` ≥ 90 % der Münzen, deutlich über `random_agent`, gepaart signifikant |
| **KW 33–34** (bis 23.08.) | Stufe 2 gelöst — der schwierige Teil | Suizidrate < 0,05 bei 300 Runden `classic`; Kisten werden zuverlässig geöffnet |
| **KW 35** (bis 30.08.) | Stufe 3 + Belohnungsablationen | schlägt `peaceful_agent` und `coin_collector_agent`; Ablationstabelle für ≥ 5 Shaping-Terme |
| **KW 36–37** (bis 13.09.) | Stufe 4, Modell B, Self-Play | Modell A schlägt `rule_based_agent`; **Entscheidung am 07.09.**, ob Modell B turnierreif wird |
| **KW 38** (bis 17.09.) | Hyperparameter, Abschlussmessung, Docker | 1000-Runden-Abschlussmessung; `docker build .` läuft; Testabgabe auf MaMPF hochgeladen |
| **21.09.** | **Code-Abgabe** | `final-project-agent-code.zip` |
| **KW 39** (bis 28.09.) | Bericht | ~12 000 Wörter, Abschnitte mit Autor markiert, Repo-URL drin |

Zwei Puffer sind bewusst eingebaut: die Testabgabe am 17.09. gibt ein Wochenende
für Absturzkorrekturen, und die Entscheidung am 07.09. verhindert, dass wir bis
zuletzt auf ein nicht konvergierendes Netz hoffen.

**Der Bericht beginnt nicht in KW 39.** Schreibt *Methods* und *Experiments*
mit, während ihr die Experimente macht — im Nachhinein rekonstruiert niemand mehr,
warum eine Entscheidung so gefallen ist. Dafür sind die Logbücher da, und dafür
schreibt `analyze.py --markdown` fertige Tabellen.

---

## 9. Nächste Schritte

1. Abschnitt 7 (Arbeitsteilung) gemeinsam ausfüllen.
2. `results/` in der `.gitignore` klären (6.7).
3. Merkmalssatz Stufe 1 implementieren, gemeinsam genutzt in `agent_code/*/features.py`
   oder einem geteilten Modul — abstimmen, wo.
4. Baseline messen, bevor irgendetwas trainiert wird:
   ```bash
   uv run python tools/evaluate.py --agents random_agent --scenario coin-heaven \
       --n-rounds 300 --label baseline_random__task1
   uv run python tools/evaluate.py --agents rule_based_agent --scenario coin-heaven \
       --n-rounds 300 --label baseline_rulebased__task1
   ```
   Das sind die Ober- und Untergrenze, gegen die alles Weitere läuft.
5. Unittest für die Explosionsgeometrie schreiben, bevor `escape_dir` gebaut wird.
