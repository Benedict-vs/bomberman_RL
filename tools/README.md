# tools/ — Messkette

Kurzreferenz. Begründung und Details: `KONZEPT.md` §6.

Einmalig für Abbildungen: `uv add matplotlib` (nur Analyse, **nicht** in die Abgabe).

---

## Vorgehen beim Messen

1. **Baseline messen, bevor ihr etwas ändert.** Ohne Vorher-Wert ist das Nachher wertlos.
2. **Eine Sache ändern**, trainieren.
3. **Nachher messen** — gleicher Seed, gleiche Rundenzahl, gleiche Gegner.
4. **Gepaart vergleichen.** Kreuzt das Konfidenzintervall die Null, ist die Verbesserung
   nicht gezeigt.
5. **Ergebnis notieren, auch wenn es negativ ist.** Kommt so in den Bericht.

Feste Regeln, sonst sind unsere Zahlen nicht vergleichbar:

- `--seed` **nie ändern** (Standard `20260731`).
- 100 Runden Schnellcheck · **300 Runden für jede berichtete Zahl** · 1000 zum Schluss.
- Dateinamen: `<person>_<modell>_<version>__<stufe>` → `maxi_q_v3__task2`

---

# 1 · `evaluate.py` — messen

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

`results/eval/<label>.csv` — eine Zeile pro **Runde und Agent**, Spalten:

| Gruppe | Spalten |
|---|---|
| Zuordnung | `round`, `seed`, `slot`, `agent`, `code` |
| Ergebnis | `score`, `coins`, `kills`, `crates`, `bombs` |
| Platzierung | `rank` (1 = bester der Runde), `won` (1 = höchster Score) |
| Überleben | `survived`, `died`, `suicides` (eigene Bombe), `killed_by_opponent` |
| Verhalten | `moves`, `invalid`, `steps`, `round_steps`, `time` |
| Laufzeit | `think_mean_ms`, `think_max_ms`, `think_over_limit` |

`results/eval/<label>.meta.json` — Git-Commit, Seed, Szenario, Abzug der `settings.py`.

---

# 2 · `analyze.py` — auswerten

## Allgemeine Form

```bash
# Modus A: Übersicht (eine oder mehrere Dateien)
uv run python tools/analyze.py <datei.csv> [weitere.csv ...] [zusätze]

# Modus B: gepaarter Vergleich zweier Läufe        <- der Normalfall
uv run python tools/analyze.py --compare <alt.csv> <neu.csv> [zusätze]

# Modus C: Ablation, viele Varianten gegen eine Baseline, EINE Metrik
uv run python tools/analyze.py --ablation <baseline.csv> <v1.csv> <v2.csv> ... \
    --metric <metrik> [zusätze]
```

Zusätze, beliebig kombinierbar:

| Flag | Wirkung |
|---|---|
| `--markdown` | Tabelle zum Kopieren in den Bericht |
| `--plot` | Abbildung nach `results/figures/`; `--plot pfad.png` für eigenen Ort |
| `--preset task1…task4` | fertiger Metriksatz für die Stufe (siehe unten) |
| `--metrics <a> <b> ...` | Auswahl der Metriken von Hand (Modus A und B) |
| `--metric <a>` | die *eine* Metrik für die Ablation (Modus C) |
| `--agent <name>` | falls nicht der Agent auf Platz 0 gemeint ist |
| `--n-boot <zahl>` | Bootstrap-Ziehungen, Standard 10000 |

## Verfügbare Metriken

| Name | Bedeutung | Richtung |
|---|---|---|
| `score` | **Primärmetrik** — Punkte der Runde (Münze 1, Kill 5) | hoch |
| `won` | Anteil Runden mit dem höchsten Score aller Agenten | hoch |
| `rank` | Platzierung in der Runde, 1 = bester | niedrig |
| `coins` | eingesammelte Münzen | hoch |
| `kills` | gesprengte Gegner | hoch |
| `suicides` | Tode durch **eigene** Bombe → Fluchtlogik kaputt | niedrig |
| `killed_by` | Tode durch **gegnerische** Bombe → Positionierung/Gefahrenwahrnehmung | niedrig |
| `died` | Todesrate gesamt (`= suicides + killed_by`) | niedrig |
| `crates` | zerstörte Kisten | hoch |
| `bombs` | gelegte Bomben | hoch |
| `survived` | Anteil überlebter Runden | hoch |
| `steps` | Schritte, die der Agent gelebt hat | hoch |
| `invalid` | ungültige Aktionen (gegen Wände laufen) | niedrig |
| `think_ms` | maximale Rechenzeit pro Zug — 0,5-s-Limit im Turnier | niedrig |

Die Spalte `verdict` in der Ausgabe: `BETTER` · `WORSE` · `no effect shown`
(letzteres, wenn das Konfidenzintervall die Null enthält).

## Welche Metriken pro Stufe — `--preset`

Statt `--metrics` von Hand aufzuzählen:

```bash
uv run python tools/analyze.py results/eval/<datei>.csv --preset task4
```

| Stufe | `--preset` | Primär | Diagnose | Worauf ihr wirklich schaut |
|---|---|---|---|---|
| **1** | `task1` | `coins` | `steps`, `invalid` | Sammelt er alle Münzen, und wie schnell? |
| **2** | `task2` | `score` | **`suicides`**, `crates`, `bombs`, `survived` | Suizidrate runter. Dazu `bombs` vs. `crates` — legt er nutzlose Bomben? |
| **3** | `task3` | `score` | `kills`, **`suicides`**, `survived` | Neue Fähigkeit `kills` hoch, `suicides` darf nicht zurückkommen |
| **4** | `task4` | `score`, `won` | `kills`, `suicides`, `killed_by`, `think_ms` | Schlägt er `rule_based`? Und *warum* stirbt er? |

**`suicides` wechselt ab Stufe 3 die Rolle, es verschwindet nicht.** Auf Stufe 2 ist es
das Fortschrittssignal (soll fallen), ab Stufe 3 der Regressionswächter (darf nicht
wieder steigen). Genau beim Lernen von Aggression vergisst ein Agent, vor der eigenen
Bombe wegzulaufen.

**Auf Stufe 4 die Todesart aufschlüsseln.** `suicides` und `killed_by` zeigen auf zwei
völlig verschiedene Baustellen: eigene Bombe → `escape_dir`/`escape_after_bomb` stimmen
nicht; fremde Bombe → der Agent stellt sich in fremde Explosionsradien. Ohne die Trennung
seht ihr nur „stirbt oft".

**`won` ist auf Stufe 4 fast wichtiger als `score`.** Das Turnier entscheidet sich gegen
die anderen Agenten: Ein Agent mit 5,0 Punkten, der 60 % der Runden anführt, ist
turniertauglicher als einer mit 5,5, der zuverlässig Zweiter wird.

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
uv run python tools/analyze.py --ablation results/eval/q_base__task2.csv \
       results/eval/q_ohne_shaping__task2.csv \
       results/eval/q_ohne_symmetrie__task2.csv \
       results/eval/q_kleine_features__task2.csv \
       --metric score --markdown --plot

# Nur die Sicherheitsmetriken anschauen
uv run python tools/analyze.py results/eval/maxi_q_v3__task2.csv \
    --metrics suicides survived invalid
```

---

# 3 · `trainlog.py` — Lernkurven

Schreibt **eine Zeile pro Episode** während des Trainings. Beantwortet „wird der Agent
über die Zeit besser", während `evaluate.py` nur den Endstand misst.

## Wo genau was hin muss

Vollständiges Gerüst — die vier markierten Blöcke sind alles, was ihr einfügt:

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

**Immer geschrieben** (Argumente von `log_episode`):

| Argument | Typ | Bedeutung |
|---|---|---|
| `episode` | int | Episodennummer |
| `score` | float | Punkte der Episode |
| `steps` | int | Schritte bis Rundenende oder Tod |
| `events` | list[str] | Ereignisliste → wird automatisch zu Zählspalten |
| `reward` | float | Summe eurer geshapten Belohnung |
| `epsilon` | float | aktuelle Explorationsrate |

Dazu automatisch `wall_clock_s` (Sekunden seit Trainingsstart).

**Aus `events` werden diese zehn Zählspalten** (ihr müsst nichts tun, nur `events`
übergeben):

`COIN_COLLECTED` · `CRATE_DESTROYED` · `COIN_FOUND` · `KILLED_OPPONENT` ·
`KILLED_SELF` · `GOT_KILLED` · `SURVIVED_ROUND` · `INVALID_ACTION` ·
`BOMB_DROPPED` · `WAITED`

Eigene Ereignisse (z. B. `MOVED_OUT_OF_BLAST`) landen **nicht** automatisch in einer
Spalte — dafür `TRACKED_EVENTS` oben in `trainlog.py` ergänzen.

**Freie Spalten** für alles Weitere: im Konstruktor `extra_columns=["td_error", "loss",
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

`--metric` akzeptiert **jede Spalte der CSV** — also auch `KILLED_SELF`, `INVALID_ACTION`,
`reward`, `epsilon` oder eure eigenen. Für Stufe 2 ist `KILLED_SELF` über die Zeit
aussagekräftiger als `score`.

## Zwei Fallstricke

**Ein `run`-Name pro Experiment.** Die Datei wird angehängt, nicht überschrieben. Startet
ihr dasselbe Training neu, beginnen die Episodennummern wieder bei 1 — dann besser einen
neuen `run`-Namen wählen.

**Pro Konfiguration 3–5 Läufe mit verschiedenen Seeds.** Eine einzelne Lernkurve ist zu
verrauscht, um etwas zu belegen.
