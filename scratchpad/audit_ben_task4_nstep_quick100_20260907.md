# Adversarialer Audit: N-step-Quick100

Datum: 2026-09-07

Geprüfte Behauptung: `nstep3_v1` ist im Quick100 gegenüber
`nstep1_control_v1` schlechter; zu kurzes Training bleibt als alternative
Erklärung möglich.

## Auditfrage

Der Audit versucht gezielt, den negativen Befund zu widerlegen: durch
Modellverwechslung, unvollständige Artefakte, abweichende Arenapaarung,
falsche Score-/Todesarithmetik, Timeouts oder ein fragiles Statistikurteil.

## Provenienz und Vollständigkeit

Beide Evaluationsdateien enthalten 400 Datenzeilen plus Kopfzeile: 100 Runden
mit vier Slots. Beide Metadateien nennen `classic`, `n_rounds=100`,
`base_seed=20260731`, `MODEL_VARIANT=trained`, dieselbe Agentenaufstellung und
denselben Callback-SHA `4413d74c...5da2fc7`.

Die Modell-SHAs aus den Metadaten stimmen mit den vorhandenen Dateien überein:

| Arm | Modell-SHA |
|---|---|
| `nstep1_control_v1` | `1fd347c16496779c36cfd02bb575920a67d81e2bd12aaf568dc3427e6ae21204` |
| `nstep3_v1` | `5cfb6f6187c6272ebb8e435f582d9a3920cef22004edafd4a49466d2051a83df` |

Die Umgebungsvariablen ordnen die Läufe korrekt zu: `n_step_return=1` für die
Kontrolle und `n_step_return=3` für den Kandidaten. Beide Trainingsmodelle
wurden über 1.000 Episoden mit Seed 11 und demselben Gegnerfeld trainiert.

## Paarung und Rohdatenchecks

Beide CSVs besitzen genau 400 eindeutige `(round, seed, slot)`-Schlüssel. Die
Schlüsselmenge des Zielagenten ist identisch; die Seeds laufen von `20260731`
bis `20260830` und die Runden von 0 bis 99. Keine Zeile überschreitet das
Zeitlimit.

Für alle 800 Zeilen gilt:

- `score = coins + 5 * kills` ohne Abweichung;
- `died` entspricht dem binären OR aus `suicides` und `killed_by_opponent`;
- im Zielagenten gibt es je 100 Zeilen pro Arm;
- gleichzeitig auftretende Suizid- und Gegnerbombenursache kommt nicht vor.

Eine erste interne Auditprüfung verwendete fälschlich die Summe der beiden
Todesursachen und vertauschte anschließend zwei CSV-Spalten. Diese Fehler lagen
in der Auditprüfung, nicht in den Dateien oder im Analysewerkzeug. Nach
Korrektur der Spaltenzuordnung sind alle Todesprüfungen fehlerfrei.

## Ergebnisreproduktion

`tools/analyze.py --compare --preset task4 --markdown` liefert:

| Metrik | Kontrolle | N-step 3 | Differenz 3 minus 1 | Urteil |
|---|---:|---:|---:|---|
| Score | 4,030 | 3,270 | `-0,760 [-1,320; -0,230]`, p=`0,0077` | WORSE |
| Win rate | 0,480 | 0,370 | `-0,110 [-0,240; +0,020]` | kein Effekt gezeigt |
| Kills | 0,150 | 0,080 | `-0,070 [-0,160; +0,010]` | kein Effekt gezeigt |
| Suicides | 0,190 | 0,350 | `+0,160 [+0,030; +0,290]`, p=`0,0207` | WORSE |
| Killed by opponent | 0,030 | 0,090 | `+0,060 [+0,000; +0,130]` | kein Effekt gezeigt |
| Think max | 0,4 ms | 0,3 ms | `-0,148 [-0,318; -0,017]` | fragil, nicht entscheidend |

Score und Suizidzeile sind nicht als fragil markiert; Bootstrap- und
Sign-Flip-Urteil stimmen dort überein. `won` wird nicht als primäre Metrik
verwendet.

## Angriff auf die Untertrainingserklärung

Beide Arme waren bereits durchgehend bei ε=`0,05`. Der 1-step-Arm zeigt über
die 100er-Fenster starke Schwankungen ohne stabilen Aufwärtstrend. Der 3-step-
Arm verbessert sich in den letzten 300 Episoden punktuell, aber ohne stabilen
Trend oder eindeutiges Plateau. Die Replaygrößen wachsen bis ungefähr 124.000
bzw. 114.000. Die 3-step-Losswerte bleiben ungefähr bei `0,20–0,22` und sinken
zuletzt leicht; wegen der unterschiedlichen Returndefinition sind sie nicht
direkt mit dem 1-step-Loss vergleichbar.

Damit ist „zu kurz trainiert“ nicht widerlegt. Der Audit widerlegt aber die
stärkere Behauptung, der Quick100-Nachteil sei durch einen Mess-, Paarungs- oder
Provenienzfehler verursacht. Die Konvergenzfrage bleibt offen.

## Endurteil

**Audit bestanden: Der negative Quick100-Befund ist technisch und statistisch
belastbar.** Das vorläufige No-Go für eine unmittelbare Full1000-Fortsetzung
bleibt bestehen. Ein längerer Lauf wäre nur als neue, vorab definierte
Untertraining-Hypothese vertretbar; er darf nicht als bloße Rettung des
Kandidaten ausgegeben werden.
