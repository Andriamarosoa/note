# Onset PCM256 — inférence compilée sans changement des poids

Cette variante complète l'expérience `ONSET_EVENT_INTEGRITY_HGB256`.
Elle ne réentraîne aucun modèle et ne change aucun seuil.

## Changements

- Parcours compilé des arbres numériques HistGradientBoosting avec Numba,
  sommes en float64 dans le même ordre que scikit-learn, sans `fastmath`.
- Exécution NumPy des MLP et de leur normalisation, avec les mêmes opérations
  float32 en place que le pipeline scikit-learn.
- Construction unique des 69 variables causales partagées par ErrorPrediction,
  FNRecovery et la nouvelle tête, avant toute sélection.
- `LowLatencyOnsetStream` accepte exactement 256 nouveaux échantillons et
  retourne la décision de ce même bloc. Les poids sont chargés et compilés
  pendant la construction, avant l'ouverture d'un flux audio.

## Vérifications préalables

- Probabilités exactement identiques sur 12465 observations archivées pour
  S701, ErrorPrediction, FNRecovery et contrôle69 (NumPy 1.26.4).
- 79 tests passent, dont les frontières d'arbres, les valeurs manquantes,
  la normalisation float32, le rejet de blocs multiples et le défaut NoteOff.
- Premier benchmark local : piste `00_BN1-129-Eb_comp`, 1000 hops dont 100
  de préchauffage, 900 mesurés. Les 28 attaques sont identiques.
  Ce premier benchmark utilise le contrôle69 avec le correctif NoteOff ;
  les mesures du nouveau HGB sont distinguées dans `resultat.md`.

| Chemin | p50 CPU | p95 CPU | p99 CPU | Hops > 5,805 ms |
|---|---:|---:|---:|---:|
| Référence instrumentée | 6,921 ms | 16,267 ms | 20,337 ms | 639/900 |
| Compilé | 3,560 ms | 5,242 ms | 7,027 ms | 25/900 |

Ces temps sont ceux de l'environnement local, hors entrée/sortie matérielle.
Le p99 dépasse encore le budget ; ce test ne certifie pas un microphone
Windows ni un fonctionnement matériel sans dépassement. Le replay parallèle
sur 80 WAV sert à vérifier les événements, pas à établir la latence matérielle.

## Reproduction

Installer `requirements-onset256.txt` avec Python 3.11 ou 3.12.
Utiliser les modèles originaux signés et l'artefact du run 39915038301.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m scripts.validate_compiled_onset256 \
  --root DONNEES_SIGNEES --previous RESULTATS_RUN_39915038301 \
  --out NOUVEAU_DOSSIER --workers 5
```

Ce contrôle doit reproduire chaque paire `(sample, pitch)` des prédictions
scellées et du deuxième replay réel. Il ne suffit pas d'obtenir le même F1.

## Portée

Les modèles sélectionnés restent ceux des cinq plis de recherche FIT80.
Cette optimisation ne transforme pas la validation par plis en preuve de
généralisation externe ni en checkpoint unique de production.
