# Audit des pertes Onset en accompagnement / polyphonie — PCM256

## Question et protocole, fixés avant la comparaison du candidat

Comparer les mêmes annotations, événements et WAV du HGB à 68,370987 % et
du candidat acoustique direct annoncé à 69,206204 %. Les 465 FN nets
supplémentaires ne constituent pas une liste de 465 événements perdus :
il faut comptabiliser séparément les anciens TP perdus et les nouvelles
attaques récupérées, puis vérifier `perdus - récupérés = 465`.

Cette série est un audit de la variante existante, sans entraînement d'une
correction, réglage de seuils sur OUTER, promotion du MIDI ou modification
d'Exact-K, Active-K, Frame Pitch, Offset et Full Note.

## Vérifications préalables

- Code de départ : `618b3f08e0af2fe8077f93245ba5ea64d4b781aa`.
- HGB de référence : TP 8050 / FP 4231 / FN 3217 ; événements scellés du dépôt.
- Candidat attendu : TP 7585 / FP 3068 / FN 3682. Une reproduction différente
  bloque l'identification à cette variante ; ne pas lui attribuer ces scores.
- 80 WAV originaux signés, 306536 blocs de 256 à 44100 Hz ; mêmes 5 compositions.
- Nouvel apprentissage : paramètres, seed, échantillonnage et politiques de
  `onset_acoustic_direct256.py` inchangés. La reproduction est nécessaire parce
  que le bilan et les poids du candidat n'étaient pas archivés dans sa branche.
- La compression des caches est sans perte, avec vérification des bits float32.
  Elle ne change ni l'audio, ni les caractéristiques, ni les sorties du modèle.
- Relecture audio indépendante : parité des caractéristiques, des événements du
  candidat et de la référence, avant toute explication des régressions.

## Unités d'analyse

Chaque annotation reçoit un identifiant stable `(piste, index de vérité terrain)`
et une classe : TP conservé, TP perdu, nouvelle récupération ou FN persistant.
L'appariement conserve le protocole original : même hauteur, un-à-un, ±80 ms,
horodatage effectif de l'émission.

La texture `comp/solo` et la polyphonie réellement annotée sont deux variables
différentes. La polyphonie est le nombre de hauteurs distinctes actives à
l'instant de l'attaque annotée ; les durées servent uniquement à l'évaluation.
Comparer aussi composition, registre et réattaque de même hauteur.

## Explications testables

1. Score direct insuffisant et absence de passage par la référence.
2. Score suffisant, mais réarmement non obtenu.
3. Proposition admissible bloquée par l'exclusion réfractaire.
4. Événement présent, mais apparié à une autre annotation de même hauteur.

Le rapport distingue le mécanisme sur tout le voisinage d'évaluation et le
motif exact au bloc où la référence émettait son ancien vrai événement.
Un faible score n'est pas une preuve de fondamentale absente ou de masquage.
Les valeurs acoustiques au premier bloc disponible sont des observations,
pas des causes physiques établies.

Trois contre-expériences fixes rejouent **toute** la piste : passage de référence
sans veto, suppression de l'exclusion réfractaire (1 hop), union exacte avec
tous les événements de référence. Rapporter les récupérations, les nouveaux
FP et les TP détruits. Ces diagnostics ne sont pas des politiques promues et
ne sont pas sélectionnés a posteriori sur OUTER.

## Reproduction

```bash
python -m pip install -r requirements-onset256-cache.txt
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_EXPERIMENT_DIR" --phase collect --cache-format npz --workers 2
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_EXPERIMENT_DIR" --phase samples
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_EXPERIMENT_DIR" --phase train --workers 2
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_EXPERIMENT_DIR" --phase replay --workers 2
python -m scripts.audit_onset_polyphony_losses256 --data "$ONSET_DATA_DIR" --experiment "$ONSET_EXPERIMENT_DIR" --out "$ONSET_AUDIT_DIR"
```

Les dossiers de sortie doivent être neufs. En cas d'interruption de collecte,
`--resume-collection` accepte seulement les pistes complètes vérifiées par SHA.
Les caches incomplets, sans métadonnées de piste, sont recalculés ; les modèles
et résultats ne sont jamais remplacés. `repack_onset_features256` ne traite
que des caches intermédiaires reproductibles, avec vérification bit à bit.

## Portée

Les 256 échantillons, zéro futur et zéro attente ajoutée concernent l'inférence.
Les voisinages d'annotation sont lus après l'inférence, pour l'évaluation.
FIT80 est déjà exposé à la recherche et aux anciens enseignants ; cette série
ne démontre pas une généralisation externe ni une latence matérielle de 5,805 ms.
