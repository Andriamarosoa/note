# Série25 — apprentissage de l'ordre inverse B→A→B→A

La série24 évalue l'effet causal d'une inversion **à poids fixés**. Cette série25 teste la nuance cruciale : si le réseau était entraîné à entendre **B avant A**, la tête forte apprendrait-elle à corriger plutôt qu'à subir ses messages ?

## Contrôle strict
- Identiques à S20 : 19 morceaux séparés un par un, folds 0/1/2/4, 335 caractéristiques acoustiques originales, proposition initiale S18, scaler adapté seulement sur les autres morceaux du fold, epochs 6, batch 512, AdamW LR .001, poids de conservation 1, auxiliaire BCE .06, dimensions de couches 128/64 pour A et 96/64 pour B.
- Le B initial voit `(signal, prior S18, prior S18)`, émet sept logit de compatibilité. A1 voit `(signal, prior, ancien état, message B0)` ; puis B1 voit A1, A2 voit B1, etc. Retour B après chaque A sauf dernier ; **même nombre de pertes BCE et d'appels A que S20**.
- Les poids sont initialisés avec le même schéma `seed=27320+100*j` que S20, mais appris de nouveau à l'ordre inversé (partage de poids B/A sur les passages). Les poids **ne sont pas partagés entre les 19 morceaux**, et l'identité du morceau évalué n'entre pas dans A ni B.
- Entraînement supervisé CE sur les quatre sorties de A, BCE pour les quatre messages de B ; perte de conservation des décisions correctes S18 **sur entraînement uniquement**.
- Quatre distributions A B-first, contrôle B neutralisé, B0 seul comme diagnostic (B seul BCE n'est pas un compte de notes officiellement calibré) ; seuils de protection fixes .10/.25/.40/.60, sans vérité dans le décodeur.
- Comparaison avec S20 original A-first **aux mêmes passages, mêmes marges**, S24 B-first à poids fixés, série18, freeze, et YourMT3+ (seulement comme métriques), par fold et par vrai K0–K6, corrections/régressions strictes, transitions et événements neutres.
- **Ne pas annoncer de promotion** en cas de perte des 2 934 corrections historiques, ou de baisse poly. Les décisions préalables du parent S18 ont été choisies post-hoc sur le corpus déjà utilisé par nos recherches ; ce n'est pas une validation sur musique inédite.

Hypothèse falsifiable : l'ordre inversé entraîné devrait, à égalité de budget d'apprentissage, augmenter les corrections B-faux→A-vrai **sans augmenter les régressions B-vrai→A-faux**, et améliorer le score poly face à S20. Tous les résultats et poids (même négatifs) seront conservés.