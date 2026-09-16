# Adversarialer Selbstaudit: Task-4 1-step versus 3-step DQN

Datum: 2026-09-07

## Auftrag, Umfang und Unabhängigkeitsgrenze

Dieser Audit versucht das vorbereitete Design `nstep1_control_v1` gegen
`nstep3_v1` vor jedem Training zu widerlegen. Geprüft werden Modellquelle,
Konfigurationsisolation, Return- und Bootstrap-Mathematik, Queue-Lebenszyklus,
Terminalbehandlung bei Tod und Survival, Augmentation, 1-step-Kompatibilität,
Replay-/Update-Confounds und freie Zielpfade.

Dies ist ausdrücklich ein **Selbstaudit durch den Implementierungsagenten** und
damit keine unabhängige zweite Meinung. Er kann technische Fehler aufdecken,
erfüllt aber die stärkere Projektregel einer unabhängig gebildeten Diagnose nur
eingeschränkt. Es wird kein Training und keine Evaluation gestartet.

## Inkrementeller Status

- Audit begonnen; Quellcode und Tests werden direkt gegen mathematisch erzeugte
  Referenzverläufe geprüft.

## Befunde

### Zwei Fehler durch den Audit gefunden und vor Training behoben

Der erste ausführbare Referenztrace widerlegte die ursprüngliche
Survival-Terminalbehandlung. Bei vier nichtterminal gemeldeten Rohschritten und
`N=3` wurde das Fenster ab Schritt 2 bereits in
`game_events_occurred()` emittiert, obwohl sein dritter Schritt die letzte
Rundenaktion war. Erst danach meldet `end_of_round()` Survival und konnte nur
noch die in der Queue verbliebenen Fenster terminal markieren. Das bereits
emittierte Fenster hätte unzulässig aus dem Zustand hinter dem Rundenende
gebootstrapped.

Die Queue behält deshalb nun bei `N>1` ein vollständiges Fenster, bis ein
folgender Schritt beweist, dass dessen Endpunkt nicht terminal war. Beim
Rundenende werden alle Fenster, die die letzte Aktion enthalten, korrekt als
terminal ausgespült. Todesübergänge tragen ihr Terminalsignal bereits beim
Append und leeren die Queue unmittelbar.

Diese Reparatur erzeugte zunächst einen zweiten Confound: Auch `N=1` wäre um
einen Schritt verzögert gespeichert worden. `N=1` wird nun ausdrücklich sofort
emittiert und reproduziert damit das bisherige Replay- und Update-Timing.

### Mathematische und ausführbare Prüfung

`scratchpad/audit_ben_task4_nstep_design.py` erzeugt unabhängig vier bekannte
Rewards und prüft sowohl Tod als auch nachträglich gemeldetes Survival. Für
beide entstehen genau vier Replayelemente mit Schrittlängen `[3,3,2,1]`,
Terminalflags `[False,True,True,True]` und Rewards
`[r1+γr2+γ²r3, r2+γr3+γ²r4, r3+γr4, r4]`. Aktionen und Startzustände bleiben
dem jeweils ältesten Rohschritt zugeordnet. Separate konstante Netze bestätigen
im echten `optimize_dqn()` numerisch `γ¹` und `γ³` im Bootstrap-Target.

Die Unit-Tests prüfen zusätzlich sofortiges Speichern für `N=1`, verkürzte
Terminalreturns, Survivalflush und Erhalt von `n_steps` durch Rotationen und
Spiegelung. Nach den Reparaturen bestehen 29 Tests.

### Konfigurationsisolation und Provenienz

Getrennte Prozessimporte bestätigen für beide Arme dieselbe Quelle
`ben_task4_mixed_kill_v1_2000ep_seed11.pt` mit SHA-256
`d50ae3d1ce80018a8a834f7df28cff3dfead5c0c99d578d7af1604a3cf0806c6`, elf
Kanäle, `γ=0,99`, Lernrate `1e-4`, identische Rewards und die Aufstellung
`peaceful_agent,rule_based_agent,rule_based_agent`. Die beabsichtigte
algorithmische Differenz ist `n_step_return=1` versus `3`; Modell- und
Laufnamen sind notwendigerweise getrennt. Sämtliche vorgesehenen Modelle,
Trainings-CSVs, Metadaten und Checkpoints sind noch abwesend.

### Verbleibende, akzeptierte Unterschiede und Grenzen

Der 3-step-Arm hält bei laufenden Episoden bis zu drei Rohübergänge außerhalb
des Replay-Buffers. Über eine vollständige Episode erzeugt er dennoch genau
einen Replayeintrag pro Rohschritt; die frühere Vermutung „zwei Einträge weniger
pro Episode“ war falsch. `MIN_REPLAY_SIZE` wird höchstens um drei beobachtete
Schritte später erreicht. Optimiert wird weiterhin höchstens einmal pro
Umgebungsschritt; beim finalen Trainingsende ausgespülte Restfenster können
nicht mehr gezogen werden. Das sind inhärente kleine Timingfolgen eines
mehrschrittigen Online-Returns, keine versteckte Reward-/Featureänderung, müssen
aber bei der Kausalformulierung genannt werden.

Die Potential-Shaping-Terme werden als Teile der Rohbelohnung ebenfalls mit
`γ^k` aggregiert. Weil sie bereits die Form `γΦ(s')−Φ(s)` besitzen, teleskopiert
dies über n Schritte korrekt. Eine 3-step-Wirkung ist trotzdem die Wirkung des
gesamten Returnverfahrens auf alle Rewardbestandteile, nicht spezifisch auf
Kills.

## Endurteil

**Nach zwei vor Training gefundenen und behobenen Fehlern technisch Go.** Die
aktuelle Implementierung bildet die beabsichtigten Returns und Terminaltargets
korrekt, bewahrt den 1-step-Kontrollpfad und isoliert die experimentelle
Änderung hinreichend. Es wurden keine Zielartefakte erzeugt.

Die Evidenz bleibt wegen Selbstaudit schwächer als eine unabhängige Prüfung.
Zulässige spätere Aussage ist ein Vergleich dieser zwei vollständigen
Trainingsverfahren; nicht zulässig wäre, einen möglichen Effekt ausschließlich
der besseren Kill-Kreditzuweisung zuzuschreiben, ohne dass Kills dies als
Mechanismus bestätigen.
