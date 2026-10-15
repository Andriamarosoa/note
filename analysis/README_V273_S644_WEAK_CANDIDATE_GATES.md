# S644 — Boucle d'audit des portes de candidats d'attaque faible

## Point de départ figé

S642 a réellement entraîné une tête d'attaque, mais le rejeu
S643 a constaté **0** tentative de rescue sur 12 GuitarSet DEV
et 5 URMP DEV, sans aucune amélioration du F1. La tête de latence
S642 a uniquement gagné **+0,57 point F1 causal GuitarSet**,
sans gain URMP ni changement du F1 historique.

Le modèle original NOTE S614, S80, S632 et les checkpoints
appris par S642 ne sont PAS modifiés. Empreintes :
- S614 : `7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`
- tête d'attaque S642 : `2f7364865803bb0cab8c564911e25574203b129169f6abf5f4ecdf14acc7f256`

## Hypothèse soumise à mesure

La politique S642 exige, pour une candidate inactive :
1. attaque existante 0,13 ≤ onset original < seuil S614 ;
2. probabilité de maintien du S614 ≥ 0,26 ;
3. score q de la nouvelle tête ≥ 0,965 ;
4. seule la meilleure candidate de la trame est élevée au seuil
   d'attaque original avant d'entrer dans le sélecteur causal S614 ;
5. le sélecteur peut encore rejeter (confirmation, réarmement,
   réfractaire, plafond de polyphonie).

Il faut compter chaque étage individuellement. Un score q élevé
n'est **pas** synonyme de NoteOn émis ; une tentative n'est
**pas** une correction.

## Expérience S644

[Workflow train-lock-dev](https://github.com/Andriamarosoa/note/actions/workflows/v273-s644-gate-anatomy-rescue.yml)

Protocole défini **avant ouverture DEV** :
- TRAIN authentique : 17 URMP + 18 GuitarSet ;
- vérifier composition disjointe de DEV et absence de sources TEST ;
- pour chaque source, calculer q du checkpoint S642 original et
  compteurs de filtrage de chacun des seuils ;
- relecture de **six variantes prédéfinies** : S614, q≥0,50,
  q≥0,70, q≥0,85, q≥0,95 et q≥0,965 original, en gardant
  les contraintes acoustiques de S642 identiques ;
- pour chaque variante : nombre d'essais, vrais NoteOn ajoutés,
  vrais positifs, faux positifs, faux négatifs et enregistrements
  corrigés/régressés ;
- verrouiller la meilleure variante **sur TRAIN uniquement**
  si elle améliore les deux corpus sans trop de régressions ;
  sinon verrouiller S614 ;
- seulement après, calculer les mêmes statistiques sur les
  17 DEV sans modifier la règle déjà choisie.

Cette grille n'entraîne **aucun nouveau poids**. Les seuils
testés ne deviennent pas automatiquement une nouvelle référence,
et les statistiques complètes DEV constituent un audit exploratoire,
pas un autre jeu de sélection valide. Toute promotion exige ensuite
une évaluation indépendante sur de nouveaux morceaux et des mesures
de causalité/latence LIVE.

## Résultats

Résultats, causes des rejets et verdict à compléter uniquement
depuis l'artefact GitHub du workflow réussi.
