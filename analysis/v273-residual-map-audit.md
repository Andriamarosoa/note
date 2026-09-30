# Carte résiduelle temps–fréquence — audit du 30 septembre 2026

La représentation passe les contrôles techniques sur les **15 952 groupes de
50 pistes** de validation interne. Elle peut servir à préparer une comparaison
native avec/sans résidu. **Aucun gain Exact K n'est établi : aucun modèle n'a
été entraîné et aucun compte existant n'a été corrigé.**

## Ce qui a changé

Le prédicteur Burg stable produit maintenant un flux de résidus : chaque bloc
de 128 échantillons est prédit à partir des 8 820 échantillons précédents.
Chaque groupe reçoit une carte de 31 instants × 64 fréquences × 4 canaux :
signal observé et résidu, pour des fenêtres de 256 et 2 048 échantillons.
Les deux durées gardent des compromis temporels/fréquentiels différents.

Les fenêtres sont alignées à droite, avec leur dernier endpoint à +2 788
échantillons, soit **63,22 ms** après l'origine. La plus longue commence à
−3 100 échantillons. Le résidu emploie encore 200 ms de passé. L'échelle
`log1p(puissance/1e-6)` est fixe, commune et sans écrêtage.

Le masque exact d'appartenance aux propositions reste une entrée séparée :
4 096 bits par groupe, accompagnés de fractions par trame. Il ne masque pas
l'audio et ne dépend d'aucune annotation. Les propositions complètes avant
troncature et leur watermark déterminent ce masque.

Cette expérience change aussi la réestimation du prédicteur et sa durée de
prévision. Ce n'est donc pas une ablation isolant la seule conservation des
axes temporel et fréquentiel. Le protocole a été écrit avant le calcul réel.

## Résultats vérifiables

| Contrôle | Résultat |
|---|---:|
| Groupes / pistes internes | 15 952 / 50 |
| Attaques attribuées avec support Hann positif et masque vrai | 9 127 / 9 127 |
| Dont avant l'origine du groupe | 2 414 |
| Dont dans l'ancienne fenêtre [0, 40 ms[ | 6 536 |
| Dont après l'ancienne fenêtre | 177 |
| Attaques couvertes hors de l'ancienne fenêtre | 2 591 (28,39 %) |
| Blocs de prévision vérifiés | 496 350 |
| Pôles instables / cartes non finies / échecs de construction | 0 / 0 / 0 |
| Module de pôle maximal, par retard | 0,999998495 |
| Recalculs sur préfixes réels indépendants | 5 / 5 identiques en float32 |
| Délai ajouté au contexte ownership sauvegardé | 0 échantillon |
| Groupes avec padding de support audio | 114 |
| Groupes touchant l'initialisation du prédicteur | 169 |
| Erreur absolue float16 maximale, unités logarithmiques | 0,00390625 |

**100 % de couverture temporelle ne signifie pas 100 % d'attaques détectées.**
Le contrôle établit que leur instant annoté figure dans une trame à poids Hann
positif et dans la région appartenant au groupe. Il n'établit ni leur
audibilité, ni leur séparation des autres notes, ni leur reconnaissance.

Les fractions d'appartenance courtes sont exactement celles de l'archive.
Les identités des lignes, les origines, les K recalculés et les trois comptes
temporels concordent avec l'audit précédent. Aucun audio du fold externe ou
de Locked12 n'a été décodé.

Les cinq contrôles de préfixe portent sur les pistes `00_*_comp.jams`, une par
composition, au groupe médian. L'audio y est arrêté au dernier échantillon
disponible et traité en chunks de 65 537 échantillons. Les cartes recalculées
sont exactement identiques à celles extraites du flux complet.

Les 15 tests unitaires (8 de carte, 7 de prédiction stable) passent. Ils couvrent
notamment le silence, les chunks arbitraires, l'arrêt au milieu d'un bloc,
la modification des suffixes futurs, les extrêmes temporels admissibles,
les watermarks insuffisants et la conservation de différences temporelles
ou fréquentielles à énergie globale égale.

La vérification indépendante rouvre les 50 caches, vérifie leurs empreintes
et énumère séparément le rayon de chaque proposition pour contrôler les
**65 339 392 bits d'appartenance**. Elle vérifie aussi 989 024 fractions par
trame, recalcule l'appartenance des annotations et leur poids Hann, ainsi que
les 169 lignes de démarrage en tenant compte de l'alignement des blocs.
Son résultat est consigné dans `verification.json`.

## Information conservée, sans score de comptage

Aucune carte résiduelle n'est constante dans le temps ou dans les fréquences.
La variance temporelle médiane est 0,849 et la variance fréquentielle médiane
1,775, en unités logarithmiques au carré. Cela établit une variation de la
représentation, pas sa pertinence pour prédire K.

Le rapport de puissance résidu/observé est calculé sur les bandes
échantillonnées, avant logarithme. Ce n'est pas une intégrale de l'énergie
audio totale et il peut dépasser 1.

| Vrai K | Groupes | Rapport médian, fenêtre courte | Rapport médian, fenêtre longue |
|---:|---:|---:|---:|
| 0 | 10 763 | 0,060 | 0,066 |
| 1 | 3 078 | 0,268 | 0,313 |
| 2 | 993 | 0,382 | 0,399 |
| 3 | 594 | 0,445 | 0,483 |
| 4 | 374 | 0,513 | 0,559 |
| 5 | 115 | 0,519 | 0,550 |
| 6 | 35 | 0,549 | 0,583 |

Ces médianes ne mesurent pas la séparation des distributions et ne constituent
pas un classifieur. Aucun seuil, signe, nouveau scalaire optimisé pour son AUC
ou classifieur diagnostique n'a été choisi sur ces données.

Le rapport dépasse 1 dans 331 groupes pour la fenêtre courte et 367 pour la
longue ; ses maxima sont 6,052 et 8,722. Un résidu élevé reste donc possible
avec des changements non musicaux ou une mauvaise prédiction. La stabilité
des pôles ne prouve pas une séparation des sources ni une petite erreur à
chaque instant.

## Limites et comparaison à préparer

- Les 64 fréquences et le zero-padding ne garantissent pas la résolution de
  notes voisines. Harmoniques, transitoires, interférences et notes étrangères
  au groupe peuvent toujours coexister dans une même carte.
- Les délais de support sauvegardés vont d'environ 63,22 à 80,11 ms ; médiane
  72,59 ms. Aucun futur supplémentaire n'est ajouté à ces entrées ownership.
  Cela ne valide pas un budget imposant 40 ms, ni une latence de production.
- L'extraction complète a pris environ 132 s avec 6 processus dans cet
  environnement. Ce temps de calcul hors ligne n'est pas un benchmark live.
- Les compositions internes ont déjà servi à plusieurs diagnostics. Une
  amélioration ultérieure sur elles ne devra pas être présentée comme une
  généralisation au jeu externe.

La suite utile est une comparaison native appariée du **K du groupe**, sans
correcteur de sortie. Conserver les fenêtres, l'échelle, l'appartenance, les
autres entrées, l'architecture, l'initialisation et le budget d'entraînement.
Le contrôle pourra dupliquer les canaux observés dans les deux emplacements
du résidu ; l'autre bras recevra les vrais résidus. Cela conserve quatre
canaux dans les deux bras et teste l'information supplémentaire du résidu.
Le protocole d'entraînement et les critères devront être figés avant lancement.

Évaluer directement Exact K, les confusions par K, les surcomptages K < 4
et les sous-comptages. La réussite de cet audit technique ne suffit pas à
autoriser une conclusion de performance. **Aucun entraînement n'est lancé
dans cette étape.**

## Reproduction et preuves

Base de code : `b0bfa2982af000f199ee31e37fa53879392f0660`.
Les scripts s'exécutent depuis la racine du dépôt avec NumPy 2.3.5,
Python 3.12.14, `PYTHONPATH=.:src`, `OPENBLAS_NUM_THREADS=1`.

```bash
python -B scripts/restore_v273_coherent_audit_inputs.py --root ../exactk_data

python -B scripts/audit_v273_coherent_real.py \
  --dataset ../exactk_data/GuitarSet \
  --geometry ../exactk_data/ownership-inputs \
  --local-only ../exactk_data/ownership-local_only \
  --with-neighbors ../exactk_data/ownership-with_neighbors \
  --config analysis/v273-native-paired-config.json \
  --output ../exactk_data/coherent-real-recomputed --workers 6

python -B -m unittest -v test.test_v273_residual_map test.test_v273_stable_prediction

python -B scripts/audit_v273_residual_map.py \
  --dataset ../exactk_data/GuitarSet \
  --geometry ../exactk_data/ownership-inputs \
  --baseline ../exactk_data/coherent-real-recomputed \
  --config analysis/v273-native-paired-config.json \
  --output ../exactk_data/residual-map-audit --workers 6

python -B scripts/verify_v273_residual_map.py \
  --root ../exactk_data/residual-map-audit \
  --dataset ../exactk_data/GuitarSet \
  --geometry ../exactk_data/ownership-inputs
```

Les répertoires de sortie doivent être absents au lancement ; réutiliser les
résultats existants pour la vérification au lieu de les écraser. L'empreinte
exacte de la baseline est figée : une autre implémentation numérique peut
nécessiter une reproduction explicite avant comparaison, jamais son omission.

Les preuves légères du dépôt se trouvent dans
`analysis/evidence/v273-residual-map/` : résumé, vérification et empreintes.
Le calcul local conserve aussi le rapport détaillé, les 15 952 lignes CSV et
les 50 caches NPZ, soit 230 332 296 octets compressés pour ces caches. Les
grandes matrices ne sont pas téléversées par le connecteur GitHub.
