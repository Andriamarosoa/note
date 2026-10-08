# V27.3 — Boucles de réduction des régressions

Protocole fixé après l'audit du croisement `parent_risk_new_correction`
(566 corrections, 518 régressions, net +48), avant d'ajuster les variantes
ci-dessous. Référence native : `freeze_local_combo`, 59 309 événements ;
7 493 événements modifiables ; folds 0, 1, 2, 4 uniquement.

## Objectif et arrêt

Réduire les régressions en conservant les corrections utiles. KEEP partout
produit zéro régression mais aussi zéro correction : ce témoin est publié,
il n'est pas l'objectif de la recherche. Sept cycles cumulatifs, définis ici,
sont exécutés sans choisir les familles à partir de leurs résultats de test.
Le minimum annoncé sera celui des politiques étudiées, jamais un minimum
universel ni une promesse de zéro erreur. Aucune promotion automatique.

0. Distribution du croisement inchangée ; coût des régressions ajustable.
1. Calibration affine positive de q et de OTHER, risque r inchangé.
2. Calibration affine positive de r et q.
3. Ajouter les transitions entre K initial et K proposé.
4. Ajouter les audits et les identités des groupes complets par destination.
5. Ajouter les 43 caractéristiques acoustiques déjà vues par le critique.
6. Ajouter les 15 caractéristiques source/cohérence également archivées,
   soit 58 au total.

Le modèle de calibration garde la factorisation cohérente
`P(initial)=r`, `P(k)=(1-r)q[k]`, `P(OTHER)=(1-r)q[OTHER]`.
Les ajouts sont des corrections linéaires régularisées des logits ; ce ne
sont pas de nouveaux réseaux profonds. La perte reste la log-vraisemblance
jointe, et les 255 groupes restent disponibles sans veto d'un constituant.
Les groupes qui donnent le même K décrivent la même action ; les descripteurs
agrègent tous ces groupes, pas un masque arbitrairement déclaré causal.

## Séparation apprentissage / choix / évaluation

Les réseaux et producteurs sont figés. Leurs sorties sur un fold ont été
produites sans entraînement sur ce fold. Pour chacun des 19 morceaux :

- évaluer uniquement ce morceau ;
- apprendre les calibrateurs sur les autres morceaux du même fold exclu
  par le réseau ; aucun autre fold de prédictions n'est utilisé ;
- choisir famille et coût sur une seconde validation par morceau, imbriquée
  à l'intérieur de ces seuls morceaux d'apprentissage ;
- ajuster ensuite la famille choisie sur tous les morceaux d'apprentissage.

Les standardisations suivent exactement les mêmes exclusions. Les identités
des événements, morceaux et enregistrements sont conservées et vérifiées.
Cette calibration utilise donc des annotations d'autres morceaux du fold
exclu par le réseau : elle **n'est pas** une évaluation laissant le fold
entier sans supervision pour le système final. Le morceau évalué est exclu
du réseau, des audits, des calibrateurs et du choix de règle. Les folds
ont déjà été examinés dans les audits précédents : aucune validation sur
données nouvelles n'est revendiquée. Fold 3 / player05 restent exclus.

## Choix d'une règle à chaque cycle

Coûts prédéfinis : `1, 1.15, 1.3, 1.5, 2, 3, 5`.
Changer seulement si `P(correction) - coût * P(régression) > 0` ; KEEP
gagne les égalités. La probabilité n'est pas modifiée par le coût.

Sur la validation interne, retenir les politiques qui :

- conservent au moins 90 % du **nombre** de corrections du croisement ;
- ont un net au moins égal au croisement ;
- ont au plus autant de régressions.

Parmi elles, minimiser les régressions ; départager par le meilleur net,
puis le plus de corrections, puis le modèle le plus simple et le coût
le plus faible. Le croisement original garantit une solution de repli.
Le seuil de 90 % n'est pas une garantie sur le morceau évalué ; publier
aussi les corrections originales réellement conservées, perdues et ajoutées.
Chaque cycle ajoute une famille aux précédentes. Une sélection interne
peut rester identique : aucune amélioration externe n'est imposée.

Pénalité L2 vers l'identité : 0,001 pour pente/interception ; 0,05 pour
les nouveaux coefficients standardisés. Pentes positives dans [0,05 ; 20],
interceptions dans [-10 ; 10]. Standardisation apprise sur les seules
données autorisées, valeurs transformées bornées à [-6 ; 6].

## Preuves et mémoire

Publier toutes les 49 politiques (7 familles × 7 coûts), les sept sorties
des cycles avec choix imbriqué, et KEEP comme témoin ; aucune suppression
selon le net. Conserver données de rejeu, paramètres, exclusions, probabilités,
décisions natives, coordonnées des cas et effets individuels.

Rapporter corrections/régressions/net, Exact-K global/poly/K0–K6, calibration,
cas sans bonne alternative, mauvais choix, synergies de groupes, résultats
par fold et morceau, et la frontière des compromis observés. Les 46
candidats antérieurs restent intacts. Les associations acoustiques servent
à décrire les erreurs ; elles ne démontrent pas leur cause physique.
