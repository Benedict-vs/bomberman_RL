# Messkette (`tools/`)

Kurzreferenz. Begründung und Details: `KONZEPT.md` §6.

Für Abbildungen einmalig `uv add matplotlib` ausführen (nur für die Analyse, nicht Teil der Abgabe).

---

## Vorgehen beim Messen

1. Baseline messen, bevor ihr etwas ändert. Ohne Vorher-Wert lässt sich das Nachher nicht
   einordnen.
2. Eine Sache ändern, trainieren.
3. Nachher messen, mit gleichem Seed, gleicher Rundenzahl und gleichen Gegnern.
4. Gepaart vergleichen. Enthält das Konfidenzintervall die Null, ist die Verbesserung
   nicht gezeigt.
5. Ergebnis notieren, auch wenn es negativ ist. Es kommt so in den Bericht.

Diese Regeln gelten für alle, sonst sind unsere Zahlen nicht vergleichbar:

- `--seed` nie ändern (Standard `20260731`).
- 100 Runden für einen Schnellcheck, **300 Runden für jede berichtete Zahl**, 1000 für die
  Schlussmessung.
- Dateinamen: `<person>_<modell>_<version>__<stufe>`, z. B. `maxi_q_v3__task2`

---

# 1. Messen mit `evaluate.py`

## Allgemeine Form

```bash
uv run python tools/evaluate.py \
    --agents   <unser_agent> [weitere ...]   # unserer immer zuerst
    --opponents <preset>                     # füllt die restlichen Plätze
    --scenario  <szenario>                   # Standard: classic
    --n-rounds  <zahl>                       # Standard: 300
    --seed      <zahl>                       # Standard: 20260731 — NICHT ändern
    --label     <dateiname>                  # ohne Endung
```

| Argument | Werte |
|---|---|
| `--agents` | Ordnernamen aus `agent_code/`, z. B. `schmaxi_coin_collector` |
| `--opponents` | `none` · `random` · `peaceful` · `coin_collector` · `mixed` · `rule_based` |
| `--scenario` | `classic` (= Turnier) · `coin-heaven` · `loot-crate` · `empty` |
| `--label` | frei; ergibt `results/eval/<label>.csv` |

Weiteres: `--out-dir`, `--log-dir`, `--quiet`.

## Aufrufe pro Stufe

```bash
# Stufe 1 — Navigation, keine Kisten, keine Gegner
uv run python tools/evaluate.py --agents <agent> --scenario coin-heaven \
    --n-rounds 300 --label <person>_<modell>_v1__task1

# Stufe 2 — Kisten und Bombenflucht, keine Gegner
uv run python tools/evaluate.py --agents <agent> --opponents none \
    --n-rounds 300 --label <person>_<modell>_v1__task2

# Stufe 3 — schwache Gegner
uv run python tools/evaluate.py --agents <agent> --opponents peaceful \
    --n-rounds 300 --label <person>_<modell>_v1__task3a
uv run python tools/evaluate.py --agents <agent> --opponents coin_collector \
    --n-rounds 300 --label <person>_<modell>_v1__task3b

# Stufe 4 — Turnierbedingungen
uv run python tools/evaluate.py --agents <agent> --opponents rule_based \
    --n-rounds 300 --label <person>_<modell>_v1__task4

# Zwei eigene Varianten direkt gegeneinander
uv run python tools/evaluate.py --agents <agent_a> <agent_b> --opponents none \
    --n-rounds 300 --label duell_a_vs_b
```

## Ausgabe

`results/eval/<label>.csv` enthält eine Zeile pro Runde und Agent mit folgenden Spalten:

| Gruppe | Spalten |
|---|---|
| Zuordnung | `round`, `seed`, `slot`, `agent`, `code` |
| Ergebnis | `score`, `coins`, `kills`, `crates`, `bombs` |
| Platzierung | `rank` (1 = bester der Runde), `won` (1 = höchster Score) |
| Überleben | `survived`, `died`, `suicides` (eigene Bombe), `killed_by_opponent` |
| Verhalten | `moves`, `invalid`, `steps`, `round_steps`, `time` |
| Laufzeit | `think_mean_ms`, `think_max_ms`, `think_over_limit` |

`results/eval/<label>.meta.json` enthält Git-Commit, Seed, Szenario und einen Abzug der `settings.py`.

---

# 2. Auswerten mit `analyze.py`

## Allgemeine Form

```bash
# Modus A: Übersicht (eine oder mehrere Dateien)
uv run python tools/analyze.py <datei.csv> [weitere.csv ...] [zusätze]

# Modus B: gepaarter Vergleich zweier Läufe        <- der Normalfall
uv run python tools/analyze.py --compare <alt.csv> <neu.csv> [zusätze]

# Modus C: Ablation — VOLLER Agent zuerst, dann je eine Variante mit
#           einer entfernten Komponente. EINE Metrik.
uv run python tools/analyze.py --ablation <voll.csv> <ohne_x.csv> <ohne_y.csv> ... \
    --metric <metrik> [zusätze]
```

Zusätze, beliebig kombinierbar:

| Flag | Wirkung |
|---|---|
| `--markdown` | Tabelle zum Kopieren in den Bericht |
| `--plot` | Abbildung nach `results/figures/`; `--plot pfad.png` für eigenen Ort |
| | Im Balkendiagramm ist der getestete Agent grün, die Gegner gedämpft blau. Standardmäßig ist das der Agent auf Platz 0 (bei `--agents` zuerst genannt) oder der mit `--agent` gewählte. |
| `--preset task1…task4` | fertiger Metriksatz für die Stufe (siehe unten) |
| `--metrics <a> <b> ...` | Auswahl der Metriken von Hand (Modus A und B) |
| `--metric <a>` | die eine Metrik für die Ablation (Modus C) |
| `--ablation-mode removal` | Standard: Baseline = voller Agent, Varianten = je eine Komponente entfernt |
| `--ablation-mode addition` | Baseline = minimaler Agent, Varianten = je eine Komponente hinzugefügt |
| `--agent <name>` | falls nicht der Agent auf Platz 0 gemeint ist |
| `--n-boot <zahl>` | Bootstrap-Ziehungen, Standard 10000 |

## Lesart der Abbildungen

Übersicht (Modus A): ein Balken pro Agent und Metrik, Fehlerbalken = 95-%-KI.
Der getestete Agent ist grün, die Gegner gedämpft blau. Ausgewählt wird der Agent
auf Platz 0 (bei `--agents` zuerst genannt) oder der mit `--agent` gewählte.

Vergleich (Modus B): ein eigenes Feld pro Metrik, jedes mit eigener Achse. Eine
gemeinsame Achse gibt es nicht, weil sich Score in Punkten, Überlebensrate als Anteil
und Rechenzeit in Millisekunden nicht sinnvoll nebeneinanderlegen lassen.

Ein häufiger Lesefehler: Dargestellt ist nicht die Differenz der beiden
Konfidenzintervalle, sondern das Konfidenzintervall der Differenz. Gerechnet wird

```
pro Runde i:   dᵢ = B(Runde i) − A(Runde i)      # beide spielen dieselbe Arena
dargestellt:   Mittelwert aller dᵢ, plus Bootstrap-KI über diese Differenzen
```

Der Unterschied ist erheblich. Zöge man die beiden Einzel-KIs voneinander ab, wäre
die Arena-Varianz wieder enthalten, die das Pairing herauskürzt. Das Intervall wäre
um ein Vielfaches breiter und damit falsch.

Pro Feld:

- Punkt = Mittelwert der Rundendifferenzen, Balken = dessen 95-%-KI
- gestrichelte Linie = keine Änderung
- grün hinterlegte Hälfte = die Richtung, die für die jeweilige Metrik besser ist
  (bei `suicides` links, bei `score` rechts)
- oben die Ausgangswerte `A … → B …`, unten Differenz, KI und Urteil

Vorschlag für die Bildunterschrift im Bericht:

> Gepaarter Vergleich über 300 Runden auf identischen Spielfeldern. Dargestellt ist
> der Mittelwert der rundenweisen Differenz (B − A) mit 95-%-Bootstrap-Konfidenzintervall
> (nicht die Differenz der Einzelintervalle). Ein Intervall, das die Null enthält, zeigt
> keinen nachgewiesenen Effekt.

Ablation (Modus C, vermutlich nicht sinnvoll, bitte nicht verwenden): eine Zeile pro
Komponente auf einer gemeinsamen Achse, weil hier alle Zeilen dieselbe Metrik zeigen.

Dargestellt ist der Beitrag der jeweiligen Komponente, nicht die Leistung des
reduzierten Agenten. Positiv heißt immer, dass die Komponente hilft, auch bei
Metriken, bei denen weniger besser ist. Bei `suicides` bedeutet `+0.16` also, dass
die Komponente die Suizidrate um 0,16 senkt.

Urteile: `MATTERS` (Weglassen hat geschadet, die Komponente verdient ihren Platz),
`HARMFUL` (der Agent war ohne sie besser), `no effect shown` (KI enthält die Null).
Bei `--ablation-mode addition` stattdessen `BETTER`/`WORSE`.

Punktfarbe: grün = gut, rot = schlecht, grau = nicht gezeigt.

## Verfügbare Metriken

| Name | Bedeutung | Richtung |
|---|---|---|
| `score` | Primärmetrik: Punkte der Runde (Münze 1, Kill 5) | hoch |
| `won` | Anteil Runden mit dem höchsten Score aller Agenten | hoch |
| `rank` | Platzierung in der Runde, 1 = bester | niedrig |
| `coins` | eingesammelte Münzen | hoch |
| `kills` | gesprengte Gegner | hoch |
| `suicides` | Tode durch eigene Bombe (Fehler in der Fluchtlogik) | niedrig |
| `killed_by` | Tode durch gegnerische Bombe (Fehler in Positionierung/Gefahrenwahrnehmung) | niedrig |
| `died` | Todesrate gesamt (`= suicides + killed_by`) | niedrig |
| `crates` | zerstörte Kisten | hoch |
| `bombs` | gelegte Bomben | hoch |
| `survived` | Anteil überlebter Runden | hoch |
| `steps` | Schritte, die der Agent gelebt hat | hoch |
| `invalid` | ungültige Aktionen (gegen Wände laufen) | niedrig |
| `think_ms` | maximale Rechenzeit pro Zug (0,5-s-Limit im Turnier) | niedrig |

Die Spalte `verdict` in der Ausgabe ist `BETTER`, `WORSE` oder `no effect shown`
(letzteres, wenn das Konfidenzintervall die Null enthält).

## Metriken pro Stufe (`--preset`)

Statt die Metriken mit `--metrics` von Hand aufzuzählen:

```bash
uv run python tools/analyze.py results/eval/<datei>.csv --preset task4
```

| Stufe | `--preset` | Primär | Diagnose | Worauf es ankommt |
|---|---|---|---|---|
| **1** | `task1` | `coins` | `steps`, `invalid` | Sammelt er alle Münzen, und wie schnell? |
| **2** | `task2` | `score` | **`suicides`**, `crates`, `bombs`, `survived` | Suizidrate senken. Dazu `bombs` vs. `crates`: Legt er nutzlose Bomben? |
| **3** | `task3` | `score` | `kills`, **`suicides`**, `survived` | `kills` als neue Fähigkeit steigt, `suicides` darf nicht wieder steigen |
| **4** | `task4` | `score`, `won` | `kills`, `suicides`, `killed_by`, `think_ms` | Schlägt er `rule_based`? Woran stirbt er? |

`suicides` bleibt ab Stufe 3 wichtig, wechselt aber die Rolle. Auf Stufe 2 ist es das
Fortschrittssignal (soll fallen), ab Stufe 3 dient es als Regressionstest (darf nicht
wieder steigen). Beim Lernen von Aggression vergisst ein Agent leicht, vor der eigenen
Bombe wegzulaufen.

Auf Stufe 4 sollte die Todesart aufgeschlüsselt werden. `suicides` und `killed_by`
weisen auf verschiedene Fehlerquellen hin. Stirbt der Agent durch die eigene Bombe,
stimmen `escape_dir`/`escape_after_bomb` nicht. Stirbt er durch eine fremde Bombe, stellt
er sich in fremde Explosionsradien. Ohne diese Trennung sieht man nur, dass er oft stirbt.

`won` ist auf Stufe 4 fast wichtiger als `score`. Im Turnier tritt man gegen die anderen
Agenten an: Ein Agent mit 5,0 Punkten, der 60 % der Runden anführt, ist für das Turnier
geeigneter als einer mit 5,5, der regelmäßig Zweiter wird.

## Typische Aufrufe

```bash
# Wie gut ist der aktuelle Stand?
uv run python tools/analyze.py results/eval/maxi_q_v3__task4.csv --preset task4

# Hat die letzte Änderung geholfen?
uv run python tools/analyze.py --compare results/eval/maxi_q_v2__task4.csv \
                                         results/eval/maxi_q_v3__task4.csv --preset task4

# Fertige Tabelle plus Abbildung für den Bericht
uv run python tools/analyze.py --compare results/eval/maxi_q_v2__task4.csv \
                                         results/eval/maxi_q_v3__task4.csv \
                                         --markdown --plot

# Welche Designentscheidungen haben tatsächlich etwas gebracht?
# ERSTE Datei = voller Agent, danach je eine Variante mit einer Komponente weniger.
uv run python tools/analyze.py --ablation results/eval/q_full__task2.csv \
       results/eval/q_ohne_shaping__task2.csv \
       results/eval/q_ohne_symmetrie__task2.csv \
       results/eval/q_ohne_escape_feature__task2.csv \
       --metric score --markdown --plot

# Nur die Sicherheitsmetriken anschauen
uv run python tools/analyze.py results/eval/maxi_q_v3__task2.csv \
    --metrics suicides survived invalid
```

---

# 3. Lernkurven mit `trainlog.py`

Schreibt während des Trainings eine Zeile pro Episode. Damit lässt sich sehen, ob der
Agent mit der Zeit besser wird, während `evaluate.py` nur den Endstand misst.

## Einbindung in `train.py`

Vollständiges Gerüst. Einzufügen sind nur die vier markierten Blöcke:

```python
# agent_code/<euer_agent>/train.py

from collections import namedtuple, deque
import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# ─── EINFÜGEN 1: Import, ganz oben zu den anderen Imports ──────────────
try:
    from tools.trainlog import TrainLogger
except ImportError:              # tools/ ist nicht Teil der Abgabe
    TrainLogger = None
# ───────────────────────────────────────────────────────────────────────


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # ─── EINFÜGEN 2: Logger anlegen, ans Ende der Funktion ─────────────
    self.trainlog = TrainLogger(
        agent="<euer_agent_ordnername>",
        run="q_v1_task1",                  # pro Experiment ÄNDERN
        hyperparams={"alpha": 0.1, "gamma": 0.95, "eps_decay": 0.9995},
        extra_columns=["td_error"],        # optional, siehe unten
    ) if TrainLogger else None
    self.episode_events = []
    self.episode_reward = 0.0
    # ───────────────────────────────────────────────────────────────────


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    ...                                    # euer bestehender Code
    reward = reward_from_events(self, events)

    # ─── EINFÜGEN 3: mitzählen, ans Ende der Funktion ──────────────────
    self.episode_events.extend(events)
    self.episode_reward += reward
    # ───────────────────────────────────────────────────────────────────


def end_of_round(self, last_game_state, last_action, events):
    ...                                    # euer bestehender Code

    # ─── EINFÜGEN 4: Zeile schreiben und zurücksetzen ──────────────────
    self.episode_events.extend(events)
    self.episode_reward += reward_from_events(self, events)
    if self.trainlog:
        self.trainlog.log_episode(
            episode=last_game_state["round"],
            score=last_game_state["self"][1],
            steps=last_game_state["step"],
            events=self.episode_events,
            reward=self.episode_reward,
            epsilon=self.epsilon,
            extra={"td_error": float(np.mean(self.losses))},   # optional
        )
    self.episode_events = []
    self.episode_reward = 0.0
    # ───────────────────────────────────────────────────────────────────
```

Woher die Werte kommen: `last_game_state["round"]` ist die Episodennummer,
`last_game_state["self"][1]` der Score, `last_game_state["step"]` die Schrittzahl.
`self.epsilon` und `self.losses` heißen bei euch so, wie ihr sie genannt habt.

## Was geloggt werden kann

Immer geschrieben werden die Argumente von `log_episode`:

| Argument | Typ | Bedeutung |
|---|---|---|
| `episode` | int | Episodennummer |
| `score` | float | Punkte der Episode |
| `steps` | int | Schritte bis Rundenende oder Tod |
| `events` | list[str] | Ereignisliste, wird automatisch in Zählspalten umgewandelt |
| `reward` | float | Summe eurer geshapten Belohnung |
| `epsilon` | float | aktuelle Explorationsrate |

Dazu automatisch `wall_clock_s` (Sekunden seit Trainingsstart).

Aus `events` entstehen diese zehn Zählspalten (dafür genügt es, `events` zu
übergeben):

`COIN_COLLECTED` · `CRATE_DESTROYED` · `COIN_FOUND` · `KILLED_OPPONENT` ·
`KILLED_SELF` · `GOT_KILLED` · `SURVIVED_ROUND` · `INVALID_ACTION` ·
`BOMB_DROPPED` · `WAITED`

Eigene Ereignisse (z. B. `MOVED_OUT_OF_BLAST`) landen nicht automatisch in einer
Spalte. Dafür `TRACKED_EVENTS` oben in `trainlog.py` ergänzen.

Freie Spalten für alles Weitere: im Konstruktor `extra_columns=["td_error", "loss",
"q_mean", "table_size"]` deklarieren, beim Loggen `extra={"td_error": 0.42, ...}`
übergeben. Typische Kandidaten: mittlerer TD-Fehler, Netzwerk-Loss, mittlerer Q-Wert,
Anzahl belegter Zustände in der Q-Tabelle, Lernrate.

## Ausgabe und Plotten

`results/train/<agent>__<run>.csv` plus `.meta.json` mit euren Hyperparametern.

```bash
# Allgemeine Form
uv run python tools/trainlog.py <datei.csv> [weitere.csv ...] \
    --metric <spalte> --window <glättung> [--labels "A" "B"] [--out pfad.png]

# Eine Kurve
uv run python tools/trainlog.py results/train/schmaxi__q_v1_task1.csv --metric score

# Zwei Läufe vergleichen
uv run python tools/trainlog.py results/train/*v1*.csv results/train/*v2*.csv \
    --metric suicides --window 200 --labels "ohne Shaping" "mit Shaping"
```

`--metric` akzeptiert jede Spalte der CSV, also auch `KILLED_SELF`, `INVALID_ACTION`,
`reward`, `epsilon` oder eure eigenen. Für Stufe 2 ist `KILLED_SELF` über die Zeit
aussagekräftiger als `score`.

## Zwei Fallstricke

Ein `run`-Name pro Experiment. Die Datei wird fortgeschrieben, nicht überschrieben.
Startet ihr dasselbe Training neu, beginnen die Episodennummern wieder bei 1. In dem Fall
besser einen neuen `run`-Namen wählen.

Pro Konfiguration 3–5 Läufe mit verschiedenen Seeds. Eine einzelne Lernkurve ist zu
verrauscht, um etwas zu belegen.

---

# 4. Checkpoint-Lernkurven mit `plot_checkpoints.py`

`trainlog.py` zeigt, was während des Trainings passiert ist: ε-greedy, gegen eine Tabelle,
die sich noch ändert. Das beantwortet, ob das Training konvergiert ist, ist laut
`README.md` aber kein Ergebnis. In E01 wurden am Trainingsende 48,2 Münzen gemessen und
1,45 in der Auswertung desselben Modells.

Dieses Skript plottet eine andere Kurve: Jeder Checkpoint wird mit `evaluate.py` bei
ε = 0 auf dem festen Arenensatz gemessen und über die Trainingsepisoden aufgetragen. Das
ist dieselbe Messung, aus der jede berichtete Zahl in `results/eval/` stammt. Deshalb kann
diese Kurve in den Bericht.

Sie zeigt, ob ein Lauf beim Abbruch schon konvergiert war. In E20 war das die zentrale
Frage: Die alte Merkmalskarte lief nach 40 000 Episoden flach (+4,51 über die letzten
60 000), die feinere stieg noch (+15,05). Ein Vergleich bei gleicher Episodenzahl
vergleicht dann zwei Punkte auf unterschiedlichen Abschnitten der Kurven.

Voraussetzung ist die Namenskonvention, die `evaluate.py --label` schreibt:

```
<präfix>_<arm>_s<seed>__ep<episoden>__<stufe>.csv
```

```bash
# Allgemeine Form
uv run python tools/plot_checkpoints.py <dateien/globs ...> \
    [--metric crates] [--reference <ref.csv>] [--out pfad.png] [--no-seeds] [--table]

# Ein Arm gegen die Baseline, mit Markdown-Tabelle für das Protokoll
uv run python tools/plot_checkpoints.py --metric crates --table \
    'results/eval/task2_crates/benedict_q_e20_dist_s*__ep*__task2.csv' \
    'results/eval/task2_crates/benedict_q_e16_c5_k03_s*__ep*__task2.csv'

# Mit Referenzagent als waagerechter Linie
uv run python tools/plot_checkpoints.py --metric crates \
    --reference results/eval/baselines/ref_rule_based_agent__task2.csv \
    'results/eval/task2_crates/benedict_q_e20_*_s*__ep*__task2.csv'
```

| Argument | Bedeutung |
|---|---|
| `--metric` | jede Spalte der Auswertungs-CSV; Standard `crates` |
| `--reference` | CSV, deren Mittelwert als waagerechte Linie eingezeichnet wird; mehrfach angebbar |
| `--no-seeds` | nur Mittelwert und Band, ohne die einzelnen Seeds |
| `--table` | druckt die Zahlen zusätzlich als Markdown-Tabelle |
| `--out` | Standard: `results/figures/curve_<metric>.png` |

Lesart der Abbildung: dicke Linie = Mittelwert über die Seeds, Band = ± 1
Standardabweichung, dünne Linien = die einzelnen Seeds. Die einzelnen Seeds werden mit
angezeigt, weil in diesem Projekt die Streuung zwischen den Seeds mehrfach selbst das
Ergebnis war und kein Rauschen (E15, E18, E20). Ein Mittelwert allein würde das verdecken.

Die Abbildungen landen in `results/figures/` und werden nicht eingecheckt, da sie aus den
CSVs erzeugt werden.
