# Versuchsprotokoll — Benedict

Ergebnisbuch, ein Eintrag pro Experiment. Bewusst getrennt von `BENEDICT.md`:
das Logbuch hält fest, *warum* ich etwas entschieden habe, hier stehen die *Zahlen*.
Die Zeilen hier werden im Bericht fast wörtlich zu Tabellenzeilen.

Regeln, an die ich mich halte:

- **Vorhersage vor der Messung.** Steht sie nicht vorher da, war es kein Experiment,
  sondern eine Beobachtung. Ich committe die Vorhersage, bevor ich messe — dann
  belegt die Git-Historie die Reihenfolge.
- **Eine Änderung pro Eintrag.** Sonst ist nicht zuzuordnen, was gewirkt hat.
- **Commit-Hash mitschreiben.** Steht in `results/eval/<label>.meta.json`. Mit festem
  Seed ist ein Lauf damit exakt reproduzierbar.
- **Negative Ergebnisse bleiben stehen.** Sie kommen so in den Bericht.
- Fester Seed `20260731`, 300 Runden für jede berichtete Zahl.

Urteil: **BESSER** · **SCHLECHTER** · **nicht gezeigt** (KI enthält die Null).

---

## E01 — Baseline: nur Wandbits

- **Frage:** Läuft die Messkette von Ende zu Ende, und wie weit kommt ein Agent,
  dessen Zustand *keine* Münzinformation enthält? Der Eintrag ist kein Versuch,
  gut zu spielen — er legt den Bezugspunkt fest, gegen den alles Weitere gemessen wird.
- **Änderung ggü. vorher:** — (Ausgangspunkt)
- **Agent:** `benedict_coin_collector` v1 · Commit `<wird nachgetragen>`
  - Merkmale: 4 Bits „Nachbarfeld blockiert?" (U/R/D/L) → Q-Tabelle 16 × 6,
    davon 11 Zeilen erreichbar
- **Training:** 1000 Runden, `coin-heaven`, keine Gegner
  - α = 0,1 · γ = 0,9 · ε = 0,2 (konstant)
  - Rewards: Münze +5 · `INVALID_ACTION` −1 · `WAITED` −0,1 · Schrittkosten −0,1
  - `KILLED_SELF` ist **nicht** in der Reward-Tabelle. Der Druck gegen `BOMB` entsteht
    allein daraus, dass die Episode endet und im Terminal-Update kein γ·max Q steht.
- **Messung:** `results/eval/benedict_q_v1__task1.csv` · coin-heaven · 300 Runden ·
  Seed 20260731 · ε = 0 (nicht im Trainingsmodus)

### Vorhersage (vor dem Lauf notiert)

1. **`invalid` ≈ 0.** Das ist das eigentliche Bestehenskriterium dieser Stufe. Die
   Wandbits stehen genau dafür im Zustand; sind die ungültigen Aktionen nicht nahe null,
   ist die Pipeline noch kaputt und alles Weitere wertlos.
2. **`steps` = 400, also die volle Runde.** Mit ε = 0 ist die Politik das Argmax der
   Tabelle, und `BOMB` war in keiner Zeile die beste Aktion. Der Agent legt also nie
   eine Bombe und kann folglich nicht sterben. (Im *Training* ist das anders: dort
   sprengt ε sich regelmäßig selbst, die Episoden brechen früh ab.)
3. **`coins` einstellig.** Die Politik ist deterministisch *und* ortsblind: gleiches
   Wandmuster → immer derselbe Zug. Damit ist die Trajektorie letztlich periodisch —
   der Agent läuft in einen kurzen Zyklus und besucht für den Rest der 400 Schritte
   dieselben paar Felder. Die Münzen, die er bekommt, sammelt er im Wesentlichen
   zufällig auf dem Weg in diesen Zyklus ein.
4. Fällt `coins` deutlich höher aus, liegt der Fehler in meinem Verständnis des
   Aufbaus, nicht im Agenten. Dann nachsehen, nicht freuen.

### Ergebnis

| Metrik | Mittelwert | 95-%-KI |
|---|---|---|
| `coins` | 1,353 | [1,153 · 1,553] |
| `steps` | 400,0 | [400,0 · 400,0] |
| `invalid` | 0,00 | [0,00 · 0,00] |
| `suicides` | 0,000 | [0,000 · 0,000] |
| `survived` | 1,000 | [1,000 · 1,000] |
| `think_max_ms` | 0,0 | — |

Alle drei Vorhersagen bestätigt, und zwar exakt: die KIs von `steps` und `invalid` sind
entartet, weil *jede einzelne* der 300 Runden 400 Schritte lief und *keine* ungültige
Aktion enthielt. Die Q-Tabelle erfüllt in allen 10 erreichbaren Zeilen mit blockierten
Richtungen das Kriterium „blockierte Richtung < jeder legale Zug".

### Der eigentliche Befund: die Zugzahl ist bimodal

`moves` ist nicht gestreut, sondern hat genau zwei Werte:

| `moves` | Runden | mittlere Münzen |
|---|---|---|
| 0 | **70** (23 %) | 0,26 |
| 400 | 230 (77 %) | 1,69 |

In 70 Runden bewegt der Agent sich **kein einziges Mal**. Die Ursache steht in der
Tabelle: Der Agent startet in einer der vier Ecken. Für die Ecke unten rechts (15,15)
sind rechts und unten Außenwand, das Merkmal ist also (U,R,D,L) = (0,1,1,0) = Zeile 6 —
und dort ist das Argmax `WAIT` (2,636) und nicht `UP` (2,455). Mit ε = 0 ist die Politik
deterministisch, das Feld ändert sich durch Warten nicht, also bleibt der Agent bis
Schritt 400 stehen. 70/300 ≈ ¼ passt genau zu „eine von vier Startecken".

Das ist kein Bug in der Implementierung, sondern die Zustandsabstraktion, die genau das
tut, was sie soll: Zeile 6 hat wenige Felder und wird selten besucht, die Schätzung ist
verrauscht, und `WAIT` liegt zufällig 0,2 über `UP`. Für ein ortsblindes Merkmal *sind*
Warten und Laufen ununterscheidbar — ohne Münzinformation gibt es keinen Grund,
das eine dem anderen vorzuziehen. Die Schrittkosten −0,1 gelten für beide gleich.

### Trainingsverlauf

`results/train/benedict_coin_collector__q_v1_task1.csv`, Kurven in `results/figures/`.

| | erste 100 Episoden | letzte 100 |
|---|---|---|
| `steps` | 39,6 | 42,1 |
| `KILLED_SELF` | 1,00 | 1,00 |
| `COIN_COLLECTED` | 3,07 | 4,08 |
| `td_error` | 1,05 | 1,33 |

Zwei Dinge, die ich ohne das Log nicht gesehen hätte:

- **Jede Trainingsepisode endet im Selbstmord.** ε = 0,2 heißt 3,3 % Chance auf `BOMB`
  pro Schritt, also im Mittel nach ~30 Schritten eine Bombe — und der Agent hat kein
  Merkmal, das ihm sagt, wo sie liegt. 1000 Trainingsrunden sind damit nur ~42 000
  Schritte, nicht 400 000. Beim *Messen* passiert das nicht, weil `BOMB` in keiner Zeile
  Argmax ist und ε = 0 gilt — daher `suicides` = 0. Der Unterschied zwischen Trainings-
  und Messverhalten ist hier extrem, und ohne die Kurve hätte ich ihn übersehen.
- **`td_error` fällt nicht, er steigt leicht.** Als Konvergenzmaß taugt der *absolute*
  TD-Fehler hier nicht: er wächst mit den Q-Werten mit, und die Belohnung ist mit ±5
  von Natur aus stark verrauscht. Für die nächste Version relativ messen oder auf
  die Änderung der Q-Tabelle zwischen Episoden umstellen.

### Urteil

**Messkette verifiziert.** `invalid` = 0 exakt, `steps` = 400 exakt, Training läuft,
Lernkurve wird geschrieben, Auswertung liefert KIs. Der Bezugspunkt steht:
**1,353 Münzen von 50**. Das ist die Zahl, die die Münzrichtung schlagen muss.

### Was ich daraus mache

1. **Nächstes Experiment (E02): Münzrichtung ins Merkmal.** Vorzeichen des Offsets zur
   nächsten Münze, 3 × 3 Werte → `|Ŝ|` = 16 × 9 = 144 Zeilen. Das ist die eine Änderung.
2. **Die Warte-Falle nicht separat reparieren.** Es ist verlockend, `WAITED` härter zu
   bestrafen oder `WAIT` aus dem Aktionsraum zu nehmen. Beides würde das Symptom
   verdecken: Sobald der Zustand eine Münzrichtung enthält, ist Laufen *nachweislich*
   besser als Warten, und die Falle verschwindet von selbst. Wenn sie das nicht tut,
   ist das eine interessante Information, die ich mir nicht wegpatchen will.
   → Für den Bericht ist der Vorher/Nachher-Vergleich von `moves` die schönere Abbildung.
3. **Trainingsrunden erhöhen.** 144 statt 16 Zeilen bei ~42 Schritten pro Episode:
   nach der Stichprobenregel (~50 Besuche je Zelle, Abdeckung ~0,2) sind das
   50 · 144 · 6 / (42 · 0,2) ≈ 5000 Runden. Sonst messe ich „zu wenig Daten" und halte
   es für „Merkmal hilft nicht".
4. Offen für später: `COIN_COLLECTED` +5 gegen die tatsächlichen +1 des Spiels ablatieren.
   Jetzt noch nicht — erst muss es überhaupt etwas geben, worauf die Belohnung wirkt.
