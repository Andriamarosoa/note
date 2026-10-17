# S648–S649 — Recherche et boucles contre les confusions harmoniques NOTE

## Verdict vérifié

L'architecture officielle NOTE/Exact-K/LIVE n'a pas été modifiée.
Les 2 séries ont évalué **10 stratégies de sélection** (dont un témoin
rejoué) sur GuitarSet réel, sans jamais charger les partitions TEST.

- S648, 5 variantes temporelles/harmoniques : CI
  https://github.com/Andriamarosoa/note/actions/runs/39001321416
- S649, 5 variantes avec support harmonique exclusif : CI de mesure
  https://github.com/Andriamarosoa/note/actions/runs/39001814004
- S648 : `S648_REAL_GUITARSET_HARMONIC_RESIDUAL_LOOPS/report.json`.
- S649 : `S649_REAL_GUITARSET_EXCLUSIVE_HARMONIC_RESIDUAL/report.json`.
- Le SHA testé pour S649 mesure est `a28d4ee4`; les commits
  ultérieurs ne modifient que le cache FFT, et leur validation
  indépendante doit être vérifiée séparément avant toute équivalence.

## Population et protocole

- 36 GuitarSet TRAIN, 9 compositions, validation croisée 5-fold
  par **composition entière** ;
- 12 enregistrements GuitarSet DEV annotés (DEV historiquement exposé) ;
- checkpoint original S614 SHA verrouillé, K natif et décodeur inchangés ;
- les exemples positifs d'entraînement sont des **propositions NoteOn
  réellement émises par le rescue** qui retrouvent un onset manqué
  par S614; toutes les fausses attaques sont négatives ;
- 761 candidats TRAIN, dont seulement **50 corrections inédites**.
  La pénurie de correctifs positifs explique la fragilité des sélections.
- Critère de CV figé : `TP - 0.5*FP`, abstention autorisée,
  aucun seuil ni branche choisis sur DEV.
- F1 évalué sur **NoteOn réels**, ±80 ms, pas sur une simple
  classification d'accord, ni sur de nouvelles notes MIDI artificielles.

## Résultats GuitarSet DEV (identiques par contrôle épinglé)

| Politique | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| S614 officiel | 998 | 585 | 582 | 63,10465% |
| Bass ×2 naïf | 1044 | 834 | 536 | 60,38172% |
| **S641 contrôle** | **1009** | **594** | **571** | **63,39931%** |
| S648 compétition65 | 1003 | 586 | 577 | 63,30073% |
| S648 nouveauté temporelle41 | 1004 | 594 | 576 | 63,18439% |
| S648 compétition + nouveauté65 | 1004 | 587 | 576 | 63,32387% |
| S648 petit modèle non linéaire | 1010 | 603 | 570 | 63,26339% |
| S649 harmoniques exclusives + 41D | 1003 | 589 | 577 | 63,24086% |
| S649 exclusives + physique65 | 1003 | 586 | 577 | 63,30073% |
| S649 exclusives + temps + physique | 1005 | 590 | 575 | 63,30709% |

S648 sélectionne sur TRAIN le petit modèle non linéaire, mais il
régresse sur DEV : +1 TP et +9 FP vs S641.
S649 choisit **le témoin S641** sur TRAIN: les politiques de
partiels exclusifs n'améliorent pas l'utilité.

### Preuves par morceau des défauts

- `Rock1-130-A_comp` : S641 détecte 124 TP / 76 FP ; le seul
  garde-fou exclusif descend à **121 TP / 76 FP**. Il perd
  trois vraies notes sans empêcher la moindre fausse attaque.
- `Funk3-112-C#_comp` : S641 183 TP /107 FP contre garde-fou
  181 TP /105 FP : une fausse note économisée coûte ici une bonne
  note environ (deux pour deux).
- `BN3-154-E_comp` : le garde-fou retire 2 FP à TP égal (36 TP).
- `Funk3-98-A_comp` : le garde-fou retire 1 FP à TP égal.

Donc un **veto harmonique fixe** serait défectueux. Dans un vrai
accord les notes indépendantes peuvent partager presque toute leur
énergie visible, en particulier les octaves. Le niveau exclusif brut
peut être petit, même pour une fondamentale réellement jouée.

## État scientifique des pistes

Les méthodes publiées pertinentes comprennent :
1. une estimation **conjointe** de plusieurs fondamentales,
   avec allocation des partiels partagés et modèle d'enveloppes
   harmoniques, plutôt qu'un veto indépendant note-par-note
   (Benetos & Dixon 2011, doi:10.1109/jstsp.2011.2162394) ;
2. la séparation / reconstruction itérative des notes fortes puis
   l'analyse du spectre **résiduel** pour détecter les autres notes
   (Delgado Castro & Szymanski 2023,
   doi:10.1121/10.0021886) ;
3. saillance harmonique qui pondère plusieurs partiels et leur
   supériorité locale plutôt que l'intensité absolue (librosa 0.11,
   `librosa.salience`).
Ces sources sont des **pistes de méthode**, pas des scores NOTE.

## Piste suivante recommandée : sélection conjointe H0/H1

Au lieu d'un correcteur qui juge chaque F0 isolément :

1. Générer depuis NOTE les candidats de notes et la distribution K
   native; figer leur provenance et leurs instants d'attaque.
2. Comparer les hypothèses `H0 = notes actuellement actives`
   et `H1 = notes actives + nouvelle fondamentale f`.
3. Ajuster les amplitudes/enveloppes des partiels avec une
   décomposition **non négative** à fenêtres strictement passées
   (par exemple NNLS régularisé ou ajustement positif projeté).
   Les partiels communs sont alloués au modèle **joint**, pas
   comptés deux fois ou automatiquement soustraits.
4. Mesurer `amélioration_de_residu(H1-H0)` avec pénalité de
   complexité et une caractéristique de preuve d'attaque f0 propre.
5. Entrainer ensuite une tête **corrections vs régressions** sur
   TRAIN, avec contexte K/notes actives et abstention.
6. Vérifier K1/K2/K3+ et les registres séparément en CV TRAIN
   par composition, puis valider DEV, puis TEST scellé et autres
   instruments seulement après gel du modèle.
7. Exiger un gain F1 stable contre S641 et aucun NoteOn
   antidaté. Bench CPU + microphone avant intégration LIVE.

L'amélioration souhaitable n'est pas un plus grand nombre de notes,
mais une attribution de l'énergie aux **sources réelles** et une
bonne décision « nouvelle attaque indépendante ou harmonique d'une
note déjà active ? ». La reconstruction sans preuve de nouveauté
peut créer une fausse note malgré un petit résidu.

**Statut : pistes de continuation identifiées ; H0/H1 joint n'a pas
encore été implémenté ni validé.**
