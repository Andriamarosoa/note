# Résultat — Onset erreur MLP69 sans rétroaction d'état

Statut : **réévaluation scientifique en cours, aucune promotion**.

## Preuves existantes

- S613 natif PCM256 : TP8099, FP4600, FN3168, F1 67,5874155 %.
- Ancien modèle ErrorPrediction MLP63 PCM256 : TP8090, FP4553, FN3177, F1 67,6704308 %.
- Ancien MLP69 avec masque appliqué directement aux probabilités d'Onset avant machine d'état : TP8089, FP4675, FN3178, F1 67,3213766 %, **régression**. 189 FP contestés, 45 TP contestés, mais nouveaux FP apparus après modification d'armement.
- Simulateur proxy événements S613 pour les mêmes poids MLP69 : TP8054, FP4411, FN3213, F1 67,8745997 % ; ce score n'était **pas** un résultat causal en flux et ne doit pas être confondu avec le nouveau score.

## Correctif à éprouver

Même réseaux 69D et même calibration figée par composition que le run en échec #39741883968. Seul le comportement de `OnsetHeadSelector` change : le verdict `P(erreur)` ne modifie plus l'état du Main ni de sa copie ; la copie reçoit toujours `p` originale et l'arbitre accepte/conteste seulement les NoteOn qu'elle a produits. Un test reproduit artificiellement un re-trigger à 2816 après remise à zéro du posterior lors d'une frame, et garantit son absence avec la nouvelle sélection.

La nouvelle mesure **réelle** doit être produite sur 80 WAV GuitarSet et 306536 frontières 256 avec la même vérité terrain. Pas de résultat nouveau tant que le CI n'est pas fini. Le programme reste Onset-only et shadow, sans ajout de latence, sans poids S708 1024, sans MIDI officiel ni modifications des cinq autres axes.
