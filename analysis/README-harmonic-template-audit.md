# Gabarits harmoniques : défaut confirmé, aucun correcteur promu

Le poids des harmoniques explique une partie de la mauvaise reconstruction.
Mais corriger cette pondération ne suffit pas à améliorer Exact-K de façon
validée : **les neuf variantes sans annotation sont rejetées par la sélection
sur FIT**. La base reste la référence.

Ce travail prolonge [l'audit acoustique](README-residual-acoustics.md) publié au
commit `0bc2c59226a3d87a44e085febd1e4d36b885d4fe`. Les données viennent du
[run 37356100423](https://github.com/Andriamarosoa/note/actions/runs/37356100423).
Les analyses sont exécutées localement avec les exports, les caches et les
archives GuitarSet vérifiés. Folds internes 0, 1, 2, 4 seulement ; aucun nouvel
entraînement neuronal. Les comparaisons sont exploratoires sur des folds déjà
examinés, pas une mesure sur un jeu indépendant.

## Diagnostic à candidats identiques

Pour les 488 cas précédents, on garde les huit candidats reconstruits et on
ajoute les fréquences annotées. Ce contrôle fournit volontairement la vérité
au reconstructeur pour séparer le problème des candidats de celui des gabarits.
Il ne peut pas être utilisé comme résultat de prédiction.

Deux changements sont isolés : forme des pics (gaussienne historique ou puissance
exacte d'une fenêtre Hann) et décroissance des hauteurs des harmoniques
(`1/sqrt(h)`, `1/h`, `1/h²`). Un autre contrôle libère les coefficients des
harmoniques à l'intérieur des groupes F0. Le [protocole](v273-harmonic-template-protocol.md)
décrit la sélection des contrôles et le cas exploratoire utilisé avant la cohorte.

| Gabarit | Triplet couvrant les 3 notes, vrais K3 / 272 | Couple couvrant les 2 notes, vrais K2 / 216 |
|---|---:|---:|
| Gaussienne, `1/sqrt(h)` historique | 0 | 29 |
| Gaussienne, `1/h` | 32 | 66 |
| Gaussienne, `1/h²` | 85 | 93 |
| Gaussienne, coefficients libres | 24 | 75 |
| Hann, `1/sqrt(h)` | 0 | 35 |
| Hann, `1/h` | 27 | 70 |
| Hann, `1/h²` | 90 | 91 |
| Hann, coefficients libres | 22 | 73 |

Correspondance univoque à 55 cents ; les groupes et horodatages restent ceux
de l'audit figé. Passer à Hann seul ne récupère aucun triplet K3 complet.
Changer seulement la décroissance, avec la gaussienne inchangée, en récupère
85. **La pondération historique contribue donc au mauvais choix des fréquences
sur ces cas.** Cela n'établit pas une cause unique de toutes les erreurs.

Libérer les coefficients ne fait pas mieux que toutes les pentes fixes. Cette
variante peut ajuster des partiels avec une fondamentale incorrecte et dispose
de davantage de paramètres : réduire le résidu n'est pas une preuve de retrouver
les notes attendues.

Le contrôle synthétique construit une somme des gabarits historiques aux
fréquences annotées, puis effectue la même recherche : **488/488 mélanges sont
retrouvés**, résidu normalisé maximal `3.54e-15`. Le solveur fonctionne dans son
propre modèle ; les désaccords avec l'audio réel ne sont pas reproduits par ce
contrôle numérique. La formule Hann est également vérifiée contre la FFT de
sinusoïdes réelles à deux phases, et le solveur par groupes contre NNLS direct.

## Test utilisable sans fréquence annotée

Réextraction de 1 666 lignes audio uniques sur 122 enregistrements. On compare
les neuf combinaisons des pentes de saillance `s` et de gabarit `t` parmi
0,5 / 1 / 2. La grille F0, les huit candidats, la suppression des voisins,
la gaussienne, les fenêtres, le compteur de base et le routage B_low restent
figés. Les seules entrées de la LR sont les deux résidus couple/triplet.

La validation concerne toujours **845 lignes d'action valides**, tous vrais K
confondus. Les bilans suivants sont les variantes fixes, toutes publiées pour
transparence ; **aucune n'est choisie à partir de ces résultats VAL**.

| Pente saillance `s` | Pente gabarit `t` | Fold 0 | Fold 1 | Fold 2 | Fold 4 | Net total |
|---|---|---:|---:|---:|---:|---:|
| 0,5 | 0,5, témoin stable | −4 | +7 | −19 | +1 | **−15** |
| 0,5 | 1 | −3 | +11 | −15 | +8 | **+1** |
| 0,5 | 2 | −1 | −4 | −22 | +10 | **−17** |
| 1 | 0,5 | −11 | +7 | −18 | +5 | **−17** |
| 1 | 1 | −4 | +5 | −21 | +15 | **−5** |
| 1 | 2 | −1 | −10 | −28 | +12 | **−27** |
| 2 | 0,5 | −5 | +5 | −19 | +5 | **−14** |
| 2 | 1 | −4 | −1 | −21 | +16 | **−10** |
| 2 | 2 | −2 | −7 | −28 | +12 | **−25** |

Le net est le nombre d'erreurs corrigées moins les prédictions exactes dégradées,
par rapport à la base. La variante à +1 corrige 141 erreurs et dégrade 140
prédictions ; ses résultats négatifs sur les folds 0 et 2 restent visibles.
Ce +1 exploratoire ne justifie pas une promotion.

### Choix uniquement sur FIT

Dans chaque fold de validation, une rotation sur les trois folds de FIT évalue
la petite LR (pondération équilibrée, seuil fixe 0,5). Le classement privilégie
le net, puis moins de régressions, puis moins d'actions. L'abstention est retenue
si aucun net FIT n'est strictement positif. Les paramètres du réseau et B_low
restent figés même pendant cette sélection interne.

| Fold de validation | Meilleur net obtenu dans la rotation FIT | Décision avant de lire VAL |
|---|---:|---|
| 0 | −62 | Abstention |
| 1 | −37 | Abstention |
| 2 | −53 | Abstention |
| 4 | −66 | Abstention |

La procédure sélectionnée conserve donc la base : **0 action, 0 correction,
0 régression, net 0**. Ce n'est pas un gain de détection. C'est le rejet documenté
de variantes dont FIT ne démontre pas l'intérêt.

## Pourquoi la meilleure reconstruction ne suffit pas

Une seconde mesure utilise les fréquences annotées seulement pour évaluer les
candidats produits depuis l'audio. Ses deux résidus concordent exactement avec
les features de l'expérience sans annotation.

| Pente de saillance | K3 dont les 3 fréquences sont parmi les 8 candidats / 272 | Meilleur nombre de triplets complets parmi les trois pentes de gabarit |
|---|---:|---:|
| 0,5 | 3 | 1 |
| 1 | 10 | 5 |
| 2 | 8 | 5 |

Même après modification de la saillance, les candidats ne contiennent pas les
trois fréquences attendues dans au moins 262/272 cas. Le test avec vérité fournie
contournait précisément cette limitation. **La sélection des candidats demeure
un défaut de représentation mesuré** ; ses mécanismes détaillés doivent encore
être isolés avant une nouvelle correction.

L'étape suivante est un audit des rangs de saillance des fondamentales attendues
et des éliminations par la suppression des voisins : distinguer une fréquence
mal classée, un pool occupé par des fréquences proches et une fondamentale
éliminée. Les annotations serviront à mesurer ces situations, jamais à remplir
le pool en prédiction. Cela ne prouve encore ni le bon correctif ni un gain futur.

## Validation et fichiers

- **20 tests réussis** : audit/rejeu précédent, attribution temporelle, noyau
  Hann, solveurs, équivalence de l'extraction et séparation des folds de FIT.
- Le calcul vectorisé reproduit les résidus du témoin stable sur les 1 666 lignes
  à `1.28e-15` près et retrouve exactement son bilan −15.
- Rejeu des 36 modèles finaux à partir des paramètres exportés : écart maximal
  des probabilités **0**, mêmes décisions et comptabilités. Aucun audio ni
  réajustement de modèle n'est nécessaire pour ce rejeu.
- [Résumé numérique](evidence/v273-harmonic-templates/summary.json),
  [bilan des neuf variantes et de la sélection](evidence/v273-harmonic-templates/decay-report.json),
  [archive des traces et modèles](evidence/v273-harmonic-templates/evidence.zip),
  [empreintes SHA-256](evidence/v273-harmonic-templates/checksums.json).

L'archive contient les 488 diagnostics détaillés, les mesures de couverture,
les features des 1 666 lignes, les scores FIT/VAL et les 36 modèles LR exportés.
Le cache spectral intermédiaire, recalculable depuis l'audio vérifié, est omis.
Les entrées audio et exports sont ceux décrits dans le README de l'audit précédent.

```bash
export PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B scripts/audit_v273_harmonic_templates.py \
  --audit output/acoustic --dataset data/GuitarSet \
  --config analysis/v273-native-paired-config.json --output output/templates
python -B scripts/audit_v273_harmonic_decay_guard.py \
  --exports inputs/exports --dataset data/GuitarSet \
  --stable output/stable --output output/decay
python -B scripts/audit_v273_decay_frequency_coverage.py \
  --cases output/acoustic/cases.jsonl --spectra output/templates/spectra.npz \
  --features output/decay/features.npz --output output/decay/frequency-coverage.json
python -B scripts/replay_v273_harmonic_decay_guard.py --input output/decay
python -B -m unittest test.test_v273_residual_audit \
  test.test_v273_residual_acoustics test.test_v273_harmonic_templates
```

Environnement : Python 3.12.14, NumPy 1.26.4, SciPy 1.17.1 et scikit-learn 1.4.2.
Cette étude concerne le chemin normal et ne mesure aucun effet du chemin compressé.

Suite : [audit du classement et de la capacité des candidats](README-candidate-ranking-audit.md).
