# Bens Logbuch

Persönliches Arbeitstagebuch: Wo ich stehengeblieben bin, woran ich gerade arbeite,
welche Ideen ich habe und warum ich mich so entschieden habe.

---

## Einträge

<!-- Neueste Einträge oben. Format:
### JJJJ-MM-TT — Kurztitel
- **Stand:** wo ich aufgehört habe
- **Gemacht:** was passiert ist
- **Entscheidung + Warum:** ...
- **Nächster Schritt / Ideen:** ...
-->

### 2026-08-21 — Bombenaktion im Task-2-Aktionsraum
- **Gemacht:** `BOMB` wurde als sechste Aktion in das DQN aufgenommen und der Ausgang des Netzes entsprechend erweitert. Die Symmetrie-Augmentierung lässt die richtungsneutralen Aktionen `BOMB` und `WAIT` unverändert; Modell-, Callback- und Augmentierungstests wurden angepasst.
- **Warum:** Der Agent muss in Task 2 selbst Bomben legen können, um Kisten zu zerstören und darin verborgene Münzen freizulegen. Die korrekte Behandlung bei der Augmentierung verhindert, dass aus einer Bombenaktion durch Drehen oder Spiegeln fälschlich eine Bewegungsaktion wird.

### 2026-08-21 — Task-2-Zustandsdarstellung
- **Gemacht:** Die CNN-Eingabe von drei auf acht Kanäle erweitert: Wände, Kisten, Münzen, eigene Position, normalisierte Bombentimer, aktive Explosionen, berechnete Bombengefahr und Bombenverfügbarkeit. Das Modell und die zugehörigen Feature- und Modelltests wurden an die neue Eingabe angepasst.
- **Warum:** Der Task-1-Zustand enthält nicht die Informationen, die der Agent zum Öffnen von Kisten und zur Flucht vor Bomben benötigt. Bombentimer und Gefahr werden getrennt dargestellt, damit das Netz sowohl die Bomben selbst als auch deren zeitlich und räumlich wirksame Explosionsbereiche erkennen kann.
