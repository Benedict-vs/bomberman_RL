# Maxis Logbuch

Persönliches Arbeitstagebuch: Wo ich stehengeblieben bin, woran ich gerade arbeite,
welche Ideen ich habe und warum ich mich so entschieden habe.

---

## Einträge

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
