# S651 — Routeur conditionnel des sélections NOTE : résultats mesurés

## Verdict

La tête de routage contextuel **ne bat pas S641** et n'est pas
promue. La politique gelée choisie sur TRAIN est explicitement
`S641_only` ; le vote majoritaire de S650 à 63,473% reste une
observation exploratoire sur DEV, pas la sélection validée par TRAIN.

## Protocole exact

- Quatre têtes séparées et gelées pour l'évaluation : B (base),
  T (nouveauté temporelle), E (support harmonique exclusif partagé),
  J (joint).
- Deux méta-routeurs entraînés : `opinions_only` (probabilités,
  désaccord, confiance) et `conditional_context` (mêmes preuves
  + onset et sustain natifs, K0–K6, registre et saillance physique).
- S614/Exact-K inchangés, nouvelles propositions passées dans
  `CausalSelection` strictement causal, aucune émission rétrodatée.
- 36 GuitarSet TRAIN avec 761 propositions, **50** corrections
  inédites. 9 compositions. Validation **imbriquée** :
  5 splits externes par composition entière ; pour chaque split,
  3 splits internes produisant les scores des têtes sans connaître
  les annotations de la composition qui sera évaluée par le routeur.
- 12 GuitarSet DEV déjà explorés dans les expériences historiques.
- TEST GuitarSet/IDMT/URMP non utilisé. LIVE officiel non modifié.
- Seuils/routeur choisis sur TRAIN ; seuils libres autorisent
  abstention ; utilité candidate pré-déclarée TP − 0,5·FP.
- Une amélioration de +2 points d'utilité sur S641 en
  validation externe TRAIN était nécessaire pour autoriser la
  promotion d'un routeur.

## Scores TRAIN nested CV

| Routeur | TP corrections | FP propositions | FN corrections | Utilité | Seuil |
|---|---:|---:|---:|---:|---:|
| S641 B original | **17** | 20 | 33 | **7,0** | 0,90 |
| Méta-routeur avis des têtes | 12 | 20 | 38 | 2,0 | 0,90 |
| Méta-routeur avec contexte acoustique/K | 12 | **8** | 38 | **8,0** | 0,92 |

Le routeur contextuel atteint seulement **+1 d'utilité** par
rapport à S641, inférieur à la marge +2 fixée. Donc la
politique officielle de S651 est *ne pas remplacer S641*.

## Scores réels DEV NoteOn (±80ms)

| Politique | TP | FP | FN | F1 | TP K3+ |
|---|---:|---:|---:|---:|---:|
| S614 officiel | 998 | 585 | 582 | 63,104647% | 550 |
| Bass Rescue sans garde | 1044 | 834 | 536 | 60,381724% | 586 |
| **S641 seul, témoin** | **1009** | **594** | **571** | **63,399309%** | **557** |
| Vote 2/3 BTE (S650) | 1007 | **586** | 573 | **63,473054%** | 554 |
| Routeur `opinions_only` | 1008 | 592 | 572 | 63,396226% | 556 |
| Routeur `conditional_context` | 1003 | **587** | 577 | 63,280757% | 553 |

Le routeur contextuel perd **six vrais TP** pour supprimer sept
faux FP vs S641, donc F1 -0,118552 point.

## Audit des causes par enregistrement

Différence du routeur contextuel vs S641 :
- `BN3-154-E_comp` : **−4 FP, 0 TP perdu** ;
- `Funk3-98-A_comp` : **−1 FP, 0 TP perdu** ;
- `Funk3-112-C#_comp` : **−2 TP et −2 FP** ;
- `Rock1-130-A_comp` : **−2 TP, 0 FP éliminé** ;
- `BN3-119-G_comp` : **−1 TP, 0 FP éliminé** ;
- `Rock1-90-C#_solo` : **−1 TP, 0 FP éliminé**.

Conclusion : le routeur sait fermer certaines fuites harmoniques
mais supprime aussi de vraies notes sur d'autres compositions.
Le problème n'est pas une simple faiblesse de seuil; les preuves
physiques de deux notes partagées se recouvrent.

## Statut technique et traçabilité

- GitHub run mesuré :
  https://github.com/Andriamarosoa/note/actions/runs/39004105566
- Le run initial a terminé **tous les audios, validations et agrégats**,
  reproduisant les trois témoins, et a écrit `report.json`.
  Il a ensuite échoué **uniquement au dernier affichage console**
  à cause d'une virgule finale en Python transformant le dictionnaire
  `report` en tuple à un élément. Les chiffres mesurés ne sont pas
  affectés. Le bug est corrigé au commit `b69be442`.
- L'artefact brut contient un tableau JSON de longueur 1
  au lieu de l'objet attendu. Une copie normalisée peut être
  produite sans changer aucun résultat; ne pas mélanger avec
  un nouveau calcul.
- Un nouveau run GitHub est requis pour attester le contrat
  de fin de workflow sur le dernier commit.

## Prochaine recherche

L'unique tête d'arbitrage globale doit être remplacée par des
**sélections à compétence locale** définies selon erreurs
individuelles par K1/K2/K3+, registre grave/médium/aigu et types
d'attaques (harmonique partagée, octave réelle, retrigger sustain,
cross-pitch leakage). Un simple méta-routeur linéaire avec 50
exemples positifs est fragile. La reconstruction conjointe
H0 (notes actives seules) / H1 (notes actives + fondamentale
candidate) reste à implémenter et auditer ; ses résidus doivent
être calculés sans anticiper de futurs échantillons audio.
