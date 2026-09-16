# Adversarialer Audit: Mixed-Kill-Fortsetzung Full1000

Datum: 2026-09-08

## Auftrag

Zu prüfen ist, ob die 7.000-Episoden-Fortsetzung gegenüber dem 2.000er
Mixed-Kill-Stand tatsächlich schlechter ist oder ob der Befund durch
unvollständige Dateien, falsche Modellwahl, Paarungsfehler, Arithmetik oder
Timingartefakte erklärt werden kann.

## Integrität

Beide Auswertungen enthalten 4.000 Datenzeilen plus Kopfzeile: 1.000 Runden,
vier Slots und identische Seeds `20260731..20261730`. Die Zielagenten-Schlüssel
`(round, seed, slot)` sind identisch. Beide Dateien haben keine Overruns. Der
neue Modell-SHA `0c190df1d39ad892705f2c0caf3498ccfda21f55759c45417fb979c90cfee62c`
stimmt mit der Modell-Datei auf Disk überein.

Für alle 8.000 Zeilen gilt `score = coins + 5*kills`; die binäre Todesvariable
stimmt mit `suicides OR killed_by_opponent` überein. Es gibt je 1.000
Zielagentenzeilen.

Die bekannte Metadatenabweichung des Trainingslaufs bleibt bestehen: Das
Trainingsskript verwendete `peaceful_agent + 2×rule_based_agent`, die alte
Trainingsmetadatei protokolliert fälschlich `3×rule_based_agent`. Das ändert
nicht die Full1000-Evaluationsaufstellung und wird offengelegt.

## Reproduziertes Ergebnis

`tools/analyze.py --compare --preset task4` ergibt 7.000 minus 2.000:

| Metrik | Differenz (95-%-CI) | Sign-Flip / Urteil |
|---|---|---|
| Score | `-0,370 [-0,574; -0,164]` | p=`0,0003`, WORSE |
| Win rate | `-0,030 [-0,070; +0,010]` | kein Effekt gezeigt |
| Kills | `-0,064 [-0,097; -0,030]` | p=`0,0001`, WORSE |
| Suicides | `-0,012 [-0,053; +0,030]` | kein Effekt gezeigt |
| Killed by opponent | `-0,001 [-0,024; +0,021]` | kein Effekt gezeigt |
| Think max | `+0,677 [+0,506; +0,856]` ms | p<`0,0001`, WORSE |

Die Scorezerlegung ist exakt `-0,050` Coins plus `5 × -0,064 = -0,320`
Killpunkte. Survival verbessert sich deskriptiv nur von `0,497` auf `0,510`.
Die maximale beobachtete Denkzeit des Zielagenten bleibt mit `18,3008 ms` unter
dem 500-ms-Limit.

## Versuch der Widerlegung

Der Befund bleibt nach Vollständigkeits-, Modell-, Paarungs-, Arithmetik- und
Timingprüfung bestehen. Score und Kills liegen klar unter null und sind nicht
fragil. Die gleichbleibende Sicherheit erklärt den Scoreverlust nicht; der
Rückgang wird fast vollständig durch verlorene Killpunkte erklärt.

## Endurteil

**Audit bestanden: Der 7.000er-Fortsetzungsstand ist gegenüber dem 2.000er
Mixed-Kill-Stand nachgewiesen schlechter auf Score und Kills.** Die Linie wird
nicht promoted und erhält kein weiteres identisches Training, keine weiteren
Seeds und keine nachträgliche Checkpointsuche. Der 2.000er Mixed-Kill-Stand
bleibt der aktuelle Task-4-Kandidat; die 7.000er-Artefakte bleiben als
Negativbefund erhalten.
