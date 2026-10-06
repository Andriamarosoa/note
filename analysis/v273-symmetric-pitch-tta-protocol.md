# Protocole — symmetric pitch TTA pour K2/K3

Date : 6 octobre 2026.

## Motivation

La cardinalité d'un accord doit rester inchangée sous une petite transposition.
Le run 37400666855 n'a exploré que les transpositions positives. Cette expérience
teste une augmentation au test-time **symétrique** afin de réduire les biais de
registre et les artefacts directionnels.

## Population

- cohorte figée de 488 cas B_low + base K3, vrais K2/K3 ;
- folds internes 0,1,2,4 ;
- fold 3 interdit ;
- chaque ligne reste évaluée par le modèle robuste de son fold.

## Vues

Pour chaque ligne :

- vues existantes +0..+6 récupérées du run 37400666855 ;
- nouvelles vues -1..-6 demi-tons calculées avec exactement le même
  `librosa.effects.pitch_shift` et le même rebuild spectral COVERED 31x64x3 ;
- candidate_set, candidate_mask et cluster_stats restent figés.

La grille totale est donc -6,-5,...,-1,0,+1,...,+6.

## Horizons

Comparer les horizons symétriques H dans {1,2,3,4,6}. Pour H, utiliser toutes
les vues de -H à +H.

## Agrégateurs figés

Quatre règles sont testées :

1. `mean_margin` :
   agir 3→2 si moyenne(P2-P3) > 0.
2. `median_margin` :
   agir 3→2 si médiane(P2-P3) > 0.
3. `mean_argmax` :
   moyenner les 7 probabilités puis agir si argmax = 2.
4. `lcb_margin` :
   agir si mean(P2-P3) - std(P2-P3)/sqrt(n_views) > 0.

Aucun seuil appris.

## Sélection cross-fold

Pour chaque fold VAL :

- évaluer toutes les paires (agrégateur,H) uniquement sur les trois folds FIT ;
- choisir le plus grand net FIT (#K2 corrigés - #K3 régressés) ;
- départage : moins de régressions, moins d'actions, plus petit H,
  ordre mean_margin, lcb_margin, median_margin, mean_argmax ;
- abstention si le meilleur net FIT <= 0 ;
- appliquer le choix inchangé au fold VAL.

Rapporter également chaque configuration fixe séparément afin de détecter si un
agrégateur robuste existe sans sélection.

## Mesures

- corrections, régressions, net ;
- par fold et total ;
- configuration choisie par rotation ;
- fraction de vues qui prédisent K2/K3 ;
- asymétrie moyenne entre +s et -s du margin P2-P3.

## Critère de progression

Progression interne si la politique FIT :

- net total > +1 (meilleur gate précédent) ;
- au moins 3/4 folds non négatifs ;
- pire fold >= -5 ;
- au moins 25 corrections conservées.

Aucune promotion automatique.
