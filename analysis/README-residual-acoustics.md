# Audit acoustique K2/K3 et correction du tri des fréquences

L'audit demandé est terminé : **488 cas horodatés**, issus de 97 enregistrements
des folds internes **0, 1, 2 et 4**. Il met en évidence une mauvaise correspondance
entre les fréquences reconstruites et les notes attendues, ainsi qu'un défaut
précis de reproductibilité du tri des candidats. La correction de ce tri est
testée ; le correcteur 3→2 reste défavorable et ne doit pas être promu.

Sources : [run figé 37356100423](https://github.com/Andriamarosoa/note/actions/runs/37356100423),
commit `cd5ed33976baa7d2d571807a6547687db1d0eaae`.
Les résultats proviennent d'une exécution locale sur les exports vérifiés de ce
run, et non d'un nouvel entraînement du réseau. Le fold 3 n'a pas été analysé.

## Ce qui est compté

Exact-K compte ici les **attaques attribuées à un groupe de candidats**. Une note
d'un groupe précédent peut encore résonner sans appartenir au K courant.
L'attribution a été recalculée avec tous les candidats d'origine : candidat le
plus proche à 20 ms au maximum, premier groupe en cas d'égalité exacte.
Les cibles obtenues correspondent aux caches d'origine pour les 97 pistes.

Tous les cas audités ont une prédiction de base K3, appartiennent à B_low et ont
deux résidus valides. Les quatre groupes sont définis par les décisions figées
du correcteur initial au seuil 0,5 :

| Groupe | Vrai K | Décision du correcteur | Cas |
|---|---:|---|---:|
| K3 dégradés | 3 | 3→2, erreur créée | 125 |
| K3 préservés | 3 | conserve 3 | 147 |
| K2 corrigés | 2 | 3→2, erreur corrigée | 108 |
| K2 restés faux | 2 | conserve 3 | 108 |

Ces 488 cas servent au diagnostic acoustique. Le bilan Exact-K du correcteur
porte sur **les 845 cas d'action valides**, y compris les autres vrais K.

## 1. Les attaques ne sont pas coupées par la fin de la fenêtre

La fenêtre postérieure mesure 2 048 échantillons à 44 100 Hz, soit **46,44 ms**.
Dans les 488 cas, aucune attaque attribuée au groupe ne commence après sa fin,
et toutes les notes attendues chevauchent cette fenêtre. Certaines attaques
précèdent son début : 49/125 K3 dégradés et 72/147 K3 préservés.

| Observation, au moins une note concernée | K3 dégradés / 125 | K3 préservés / 147 |
|---|---:|---:|
| Attaque attendue après la fenêtre | 0 | 0 |
| Note attendue sans chevauchement de la fenêtre | 0 | 0 |
| Couverture de la pondération Hann² inférieure à 10 % | 22 | 14 |
| Autre note audible dans la fenêtre | 74 | 115 |
| Nouvelle attaque d'un autre groupe dans la fenêtre | 8 | 4 |

La couverture Hann² décrit la position et la durée annotées ; elle ne mesure
pas l'énergie isolée d'une note. Les observations se recouvrent et ne constituent
pas des causes exclusives. Les résonances voisines sont aussi fréquentes chez
les K3 préservés : leur présence seule n'explique pas les régressions.

Un contrôle plus comparable associe chaque K3 dégradé à un K3 préservé de la même
piste, en privilégiant les hauteurs annotées proches, puis le temps. Il trouve
85 paires, avec réutilisation de 46 témoins distincts ; 40 erreurs restent sans
témoin. La différence de faible couverture ne persiste pas dans ce contrôle.
Allonger la fenêtre n'est donc pas une correction démontrée par cet audit.

## 2. Les fréquences reconstruites représentent mal les notes du groupe

Le code utilise le spectre positif `max(puissance_post − puissance_pre, 0)`,
une grille de 240 fondamentales entre 65 et 1 800 Hz, huit candidats et des
gabarits de dix harmoniques. Il sélectionne le meilleur couple et le meilleur
triplet par erreur de reconstruction non négative. Chaque harmonique du gabarit
a une forme gaussienne de largeur fixe 18 Hz et un poids fixe `1/sqrt(h)`.

La comparaison aux annotations est **univoque**, avec une tolérance de 55 cents :
un candidat ne peut pas couvrir deux notes attendues.

| Groupe | Toutes les notes attendues parmi les 8 candidats | Toutes dans le triplet choisi |
|---|---:|---:|
| K3 dégradés | 1/125 | 0/125 |
| K3 préservés | 2/147 | 0/147 |
| K2 corrigés | 5/108 | 3/108 |
| K2 restés faux | 30/108 | 23/108 |

Sur les **272 vrais K3, seuls 3** ont leurs trois fréquences représentées parmi
les huit candidats. Aucun triplet choisi ne représente les trois notes.
Cela concerne aussi les cas où le compteur garde correctement K3 : ces
fréquences ne sont donc pas une transcription fiable des notes du groupe.
Ce constat ne signifie pas que le réseau principal échoue sur tous ces cas.

Le contrôle avec les contours de pitch par corde donne toujours 3/272 cas
entièrement couverts. Les contours sont disponibles pour 809 des 816 notes ;
l'écart médian avec les annotations est de 3,08 cents chez les K3 dégradés et
de 2,02 cents chez les préservés. Un simple écart d'accordage n'explique pas
la très faible couverture.

### Test avec les fréquences attendues fournies au reconstructeur

On ajoute les trois fréquences annotées aux huit candidats, puis on conserve
exactement les mêmes gabarits et le même critère de sélection du triplet.
**Le meilleur triplet ne couvre toujours les trois notes dans aucun des
272 cas.** Ce test utilise la vérité annotée uniquement comme diagnostic ;
il n'est pas une méthode utilisable en prédiction.

Cela montre qu'ajouter des candidats ne suffit pas avec la représentation et
le critère actuels. Un petit résidu spectral n'est pas une preuve qu'on a
retrouvé les trois attaques attendues. L'audit ne sépare pas encore les effets
de la rigidité des gabarits, des transitoires et des autres notes audibles.

Retirer simplement la soustraction du spectre précédent n'améliore pas non plus
la couverture moyenne : variation de −0,032 note chez les K3 dégradés et
−0,116 chez les préservés. C'est un diagnostic de représentation, pas une mesure
d'Exact-K d'un nouveau correcteur.

### Exemple horodaté

Cas `7636`, piste `00_Jazz3-150-C_comp.jams`, début **18,79948 s** : vrai K3,
prédiction de base 3, abaissée à tort à 2.

| Fréquence annotée | Attaque par rapport au début de fenêtre |
|---:|---:|
| 164,93 Hz | −1,59 ms |
| 248,01 Hz | +12,00 ms |
| 393,35 Hz | +21,04 ms |

Le triplet retenu est **84,64 / 143,52 / 162,64 Hz**. Il couvre une seule des
trois fréquences attendues à 55 cents près. Les deux autres ne figurent même
pas parmi les huit candidats de ce cas.

Les registres low/mid/high de l'audit précédent étaient calculés à partir de
ces triplets estimés. Ils ne doivent pas être interprétés comme le registre
réel des notes. Les hauteurs annotées figurent désormais dans chaque cas.

## 3. Défaut de tri confirmé et corrigé

Des fondamentales voisines peuvent recevoir **exactement le même score** parce
que la saillance lit les mêmes pics FFT. Le tri NumPy sans option stable pouvait
changer leur ordre ; la suppression des candidats trop proches conservait alors
une fréquence différente, ce qui modifiait les résidus.

Exemple `21434`, piste `01_Jazz1-130-D_comp.jams`, à 7,22757 s : 73,6594 et
74,6901 Hz ont le même score `0.06921738788028282`. Deux ordres valides produisent
un résidu triplet de `0.5470743651642915` ou `0.5505829519844698`. Ce dernier
reproduit l'export figé. L'audio est identique.

La correction utilise `np.argsort(-score, kind="stable")` : en cas d'égalité,
l'ordre croissant de la grille est conservé avant la suppression des voisins.
Le reste de l'extraction reste identique. Un test couvre spécifiquement ce cas.

Pour auditer les anciens exports, 10/488 lignes ont nécessité un autre ordre
des égalités reproduisant les deux résidus et la médiane F0 exportés. L'écart
maximal final des résidus est `6.44e-15`. Ces trois valeurs ne permettent pas de
prouver l'identité des huit candidats historiques : cette limite est conservée
dans le rapport. Quatorze lignes physiques avaient aussi des résidus différents
selon leurs apparitions FIT/VAL dans les exports d'origine.

## 4. Effet mesuré de la seule correction du tri

Réextraction de 1 666 lignes audio uniques, sur 122 pistes internes. Les
prédictions du réseau et le routage B_low sont figés. La même régression
logistique à deux résidus est réajustée sur FIT seulement, avec pondération
équilibrée et seuil 0,5 inchangés. Le réajustement témoin sur les anciennes
features reproduit les probabilités exportées à `2.78e-16` près.

| Fold interne | Bilan Exact-K initial | Avec tri stable |
|---|---:|---:|
| 0 | −4 | −4 |
| 1 | +6 | +7 |
| 2 | −20 | −19 |
| 4 | +1 | +1 |
| **Total** | **−17** | **−15** |

| Ensemble des 845 cas d'action valides | Initial | Tri stable |
|---|---:|---:|
| Actions 3→2 | 427 | 430 |
| Erreurs corrigées | 108 | 109 |
| Prédictions exactes dégradées | 125 | 124 |
| Actions sur d'autres vrais K, restant fausses | 194 | 197 |

Le tri stable corrige la reproductibilité. **Le correcteur perd encore
15 prédictions exactes par rapport au compteur de base** ; aucun modèle n'est
promu. Cette comparaison interne est exploratoire sur des folds déjà examinés,
pas une validation indépendante d'un gain généralisable.

## Suite justifiée par les résultats

La priorité est de revoir la représentation et la sélection des composantes
harmoniques, puis de vérifier leur correspondance avec les attaques attendues
avant de réentraîner une tête de comptage. Les candidats fournis par la vérité
doivent rester un contrôle de diagnostic, jamais une entrée d'inférence.

L'architecture normal + compressé reste une question distincte : cet audit
porte sur les résidus du chemin audio normal existant. Il ne mesure pas l'effet
d'une compression ×4 ou ×6 et ne permet pas d'attribuer les erreurs historiques
de compression à la même cause.

## Fichiers et reproduction

- [Résumé numérique](evidence/v273-residual-acoustics/summary.json).
- [Rapport de l'intervention de tri](evidence/v273-residual-acoustics/stable-pool-report.json).
- [Archive complète](evidence/v273-residual-acoustics/evidence.zip) : 488 traces
  `acoustic/cases.jsonl`, rapports globaux et par fold, témoins appariés,
  272 diagnostics avec annotations, features/scores et paramètres des LR du test de tri.
- [Empreintes](evidence/v273-residual-acoustics/checksums.json).

Environnement numérique exécuté : Python 3.12.14, NumPy 1.26.4, SciPy 1.17.1,
scikit-learn 1.4.2. Les versions numériques correspondent à celles des exports
sources ; leur Python était 3.11.16. Aucun TensorFlow n'est nécessaire pour ces
trois nouveaux scripts.

Préparer les quatre exports du run source sous `inputs/exports/fold-{0,1,2,4}`,
les dix ZIP `training-part-*.zip` de la release
[v273-window-pair-36351028493](https://github.com/Andriamarosoa/note/releases/tag/v273-window-pair-36351028493)
sous `inputs/parts`, et les ZIP officiels `annotation.zip` et
`audio_mono-pickup_mix.zip` de [GuitarSet](https://zenodo.org/records/3371780)
sous `data/GuitarSet`. L'audit vérifie leurs MD5, la config et les SHA-256 des
caches NPZ réellement utilisés. Lancer depuis la racine du dépôt :

```bash
export PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B scripts/audit_v273_internal_residual_acoustics.py \
  --exports inputs/exports --dataset data/GuitarSet --parts inputs/parts \
  --config analysis/v273-native-paired-config.json --output output/acoustic
python -B scripts/probe_v273_residual_dictionary.py \
  --cases output/acoustic/cases.jsonl --dataset data/GuitarSet \
  --output output/acoustic/dictionary-probe.json
python -B scripts/audit_v273_residual_stable_pool.py \
  --exports inputs/exports --dataset data/GuitarSet --output output/stable
python -B -m unittest test.test_v273_residual_audit test.test_v273_residual_acoustics -v
```

Le contrôle CI léger vérifie les 14 tests, la syntaxe et les empreintes des
résultats conservés. Il n'exécute pas à nouveau l'audit audio. L'ancien workflow
`v273-failure-clustering`, qui lit des résultats du fold 3, exige désormais un
fichier de lancement dédié ou un déclenchement manuel au lieu de tourner à
chaque push sur cette branche.

Suite : [contrôles des gabarits et test des pentes sans annotation](README-harmonic-template-audit.md).
