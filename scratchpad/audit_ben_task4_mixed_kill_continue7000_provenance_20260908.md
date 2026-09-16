# Provenienz-Audit: Mixed-Kill-Fortsetzung 7000

Datum: 2026-09-08

Der Lauf `mixed_kill_continue7000_v1` ist technisch vollständig: 5.000
Trainingszeilen plus Kopfzeile, Checkpoints 100 bis 5.000, Endmodell und
Metadatei sind vorhanden. Das Endmodell lädt den 2.000-Episoden-Mixed-Kill-
Checkpoint; die Ausgabe ist als 7.000-Episoden-Gesamtstand benannt.

Es gibt eine Metadatenabweichung. Das Startskript übergab tatsächlich
`peaceful_agent, rule_based_agent, rule_based_agent`, während die vorhandene
Metadatei `training_opponents=rule_based_agent,rule_based_agent,rule_based_agent`
schreibt. Ursache war eine falsche Set-Zuordnung in `train.py`; sie wurde nach
dem Lauf korrigiert. Die tatsächlich ausgeführte CLI ist im Startskript sichtbar.

Die Rohdaten und Metadaten werden nicht nachträglich umgeschrieben. Das Modell
bleibt als technischer Trainingslauf erhalten, ist aber bis zur korrekten
greedy Auswertung nicht als vollständig provenanceierter Promotion-Kandidat zu
behandeln. Die Abweichung muss in jeder Ergebnisdarstellung offengelegt werden.
