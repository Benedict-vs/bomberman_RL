# Benedicts Logbuch

Persönliches Arbeitstagebuch: Wo ich stehengeblieben bin, woran ich gerade arbeite,
welche Ideen ich habe und warum ich mich so entschieden habe.

---

## Einträge

### 2026-07-31 — Setup steht, Agent ist noch die blanke Vorlage

- **Stand:** Repo-Gerüst ist fertig: uv-Environment (Python 3.12, `pyproject.toml` + `uv.lock`),
  Framework im Root, `AGENTS.md`/`CLAUDE.md` als Projektspec, `help/` mit Regeln, Deadlines und
  Terminal-Kommandos. Heute dazu: pro Person ein Logbuch (`BENEDICT.md`, `MAXI.md`, `BEN.md`),
  referenziert in `AGENTS.md` und `CLAUDE.md`.
- **Gemacht:** Jeder hat aus `tpl_agent` einen eigenen Ordner kopiert
  (`benedict_coin_collector`, `ben_coin_collector`, `schmaxi_coin_collector`). Inhaltlich ist
  meiner noch 1:1 die Vorlage — `act()` würfelt aus einer festen Wahrscheinlichkeitsverteilung,
  `state_to_features()` ist ein Stub (`channels.append(...)`), `train.py` sammelt Transitions in
  einer Deque der Länge 3 und pickelt nur `self.model`. Also: **noch kein Lernen, kein Feature.**
- **Entscheidung + Warum:**
  - Getrennte Agent-Ordner pro Person, damit wir parallel experimentieren können, ohne uns die
    Modelle zu überschreiben. Zusammengeführt wird später auf den besten Ansatz — laut Aufgabe
    brauchen wir ohnehin **zwei** beschriebene Modelle im Report.
  - Erst Task 1 (`coin-heaven`, keine Kisten/Gegner) statt direkt `classic`: kleiner
    Zustandsraum, schnelles Feedback, und wir brauchen von Anfang an saubere Metriken —
    Wissenschaftlichkeit ist das Hauptbewertungskriterium.
  - Kein Deep Learning zum Start. Die Projektbeschreibung sagt, einfache, gut getunte Modelle
    haben historisch gewonnen; also erstmal tabellarisches/lineares Q-Learning auf wenigen,
    gut überlegten Features.
- **Nächster Schritt / Ideen:**
  1. `state_to_features()` für `coin-heaven` bauen: Richtung zur nächsten Münze (BFS, one-hot),
     freie/blockierte Nachbarfelder. Bewusst *keine* Feature, die die beste Aktion zurückgibt —
     das ist explizit verboten.
  2. Q-Learning mit Reward-Shaping: näher an Münze `+`, weiter weg `−` (jede Belohnung braucht
     ihr Gegenstück), `INVALID_ACTION` bestrafen.
  3. Metrik festlegen, bevor trainiert wird: Münzen pro Runde und Schritte bis zur letzten Münze,
     gemittelt über N Runden mit festem `--seed`, als Baseline gegen `random_agent`.
  4. Symmetrien (4× Rotation, Spiegelung) für Daten-Augmentation prüfen — später, erst wenn die
     Basis-Pipeline läuft.
  5. Offen: gemeinsames Auswertungsskript für `results/`, damit wir drei die Metriken vergleichbar
     loggen. Sollte ich mit Maxi und Ben absprechen.

<!-- Neueste Einträge oben. Format:
### JJJJ-MM-TT — Kurztitel
- **Stand:** wo ich aufgehört habe
- **Gemacht:** was passiert ist
- **Entscheidung + Warum:** ...
- **Nächster Schritt / Ideen:** ...
-->
