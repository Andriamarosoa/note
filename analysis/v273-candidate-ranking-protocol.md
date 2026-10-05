# Audit du classement des candidats — 5 octobre 2026

Point de départ : `aad8dfb`, 488 cas figés des folds 0, 1, 2 et 4. Les trois
fréquences attendues figurent parmi les huit candidats dans seulement 3, 10 ou
8 des 272 vrais K3 selon la pente de saillance. Le fold 3 reste exclu.

Avant de proposer un changement du pool, mesurer pour les trois pentes déjà
auditées (0,5 / 1 / 2), sans réajuster de modèle :

1. Pour chaque note annotée, rang brut du premier point de grille à 55 cents,
   rang après NMS et éventuels points ayant supprimé les voisins de la note.
   Poursuivre la NMS jusqu'à épuisement de la grille, sans arrêt à huit.
2. Couverture univoque des notes dans les 8, 16, 32, 64 et tous les candidats
   après NMS ; comparer au classement brut avant NMS. Une couverture à pool
   élargi n'est pas un gain d'Exact-K et ne sélectionne pas sa taille.
3. Distinguer hors-grille, disparition lors de la NMS, présence après la
   huitième place et concurrence univoque entre notes voisines. Les catégories
   par note décrivent des mécanismes de perte de couverture, pas la cause
   exclusive du surcomptage du compteur.
4. Tracer les fréquences du pool, scores, rangs et bins FFT dominants. Vérifier
   la reproduction exacte de la couverture déjà publiée des neuf variantes.

Les spectres sont les spectres positifs post-moins-pré du chemin normal,
recalculés dans l'audit précédent. Les annotations ne servent qu'au diagnostic.
Conserver les populations et la référence. Tout changement ultérieur doit
avoir un motif précis, des contrôles et une comparaison sans annotations avec
sélection exclusivement sur FIT, selon le protocole des gabarits.

## Contrôle de capacité décidé après le diagnostic

Le bras historique perd 499/816 notes K3 après la huitième place, contre cinq
entièrement supprimées par NMS. La couverture complète passe de 3/272 à 267/272
avec les 64 premiers candidats NMS. La grille complète couvre les 272 cas.
Tester donc un plan factoriel limité : pool 8 ou 64, gabarit gaussien de pente
0,5 ou 2. La saillance historique de pente 0,5 reste figée. Les deux facteurs
correspondent aux deux défauts documentés ; ni la fenêtre, ni le routage, ni la
population, ni la LR à deux résidus et seuil 0,5 ne changent.

Les bras pool-8 ont déjà été mesurés ; ils servent de contrôles exacts. Ne pas
choisir un bras à partir de sa VAL. La sélection sur FIT tourne uniquement sur
les autres folds internes et s'abstient en l'absence de net strictement positif,
avec le même départage que l'étape précédente. Rapporter les quatre bras fixes,
la politique sélectionnée et la couverture diagnostique des triplets. Aucun
gain de couverture ne suffit à promouvoir un correcteur. Les folds internes
sont déjà examinés ; ne pas les présenter comme un test indépendant intact.
