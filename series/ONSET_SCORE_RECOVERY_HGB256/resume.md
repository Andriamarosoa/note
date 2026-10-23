# Onset PCM256 — priorité aux scores

Consigne utilisateur du 23 octobre 2026 : améliorer les scores ; les mesures
de performance sont reportées. La décision reste au bloc courant de 256
échantillons, sans audio futur ni confirmation sur un bloc supplémentaire.

## Référence figée

Le run 39915038301 a validé TP 8012 / FP 4264 / FN 3255, F1 68,062694 %,
sur 80 WAV et 306536 blocs. Sa tête de sélection principale est conservée.

## Première hypothèse écartée

`onset_robust_cal256.py` entraîne 85 HGB avec quatre compositions de calibration
interne par pli. Résultat sur les événements archivés : TP 8031 / FP 4434 /
FN 3236, F1 67,680769 %. Cette variante est écartée avant un nouveau replay
audio : aucun gain réel n'est revendiqué pour elle. Rapport intégral conservé.

## Nouvelle récupération d'attaques

- Propositions causales des trois politiques existantes : sensitive, moderate,
  cautious. Source signée : run 39754613388, artefact 12416445957, archive SHA256
  `69d4bf995425ec6efa6341e15669931252a930f25a6c193c29a34673ae279f53`.
- Seules les propositions admissibles avec les attaques natives passées
  entrent dans l'apprentissage. Aucun accès aux attaques natives futures.
- Les cibles sont recalculées après l'inférence, par rapport aux attaques
  réellement retenues par la tête principale figée. Les anciennes récupérations
  ne sont pas retirées des cibles : le nouvel expert doit aussi pouvoir les garder.
- Deux capacités HGB, 15 et 31 feuilles, paramètres fixés dans la
  préinscription ; 120 modèles internes et au maximum 5 modèles finaux.
- Pour chaque OUTER, quatre compositions internes servent chacune de CAL à
  un modèle entraîné sur les trois autres. Politique, capacité et seuil sont
  choisis au meilleur F1 micro agrégé des 64 pistes internes. Aucun plafond
  arbitraire de perte de TP, de FP ajoutés ni de temps CPU.
- L'ancien expert de récupération est un candidat explicite de repli.
- Le modèle retenu est réentraîné sur les quatre compositions internes, puis
  les poids et le seuil sont sauvegardés avant toute prédiction OUTER.

Le calcul sur événements reproduit exactement les anciennes récupérations
du replay réel pour les 80 pistes. Le calcul de score garde le même appariement
hongrois un-à-un, même hauteur et tolérance ±80 ms. Cette tolérance intervient
après l'inférence et n'introduit aucune attente dans le flux.

## Validation

Un gain sur les événements archivés doit être confirmé par un nouveau passage
des 80 WAV originaux. Chaque paire `(sample, pitch)` doit être identique aux
prédictions figées. La validation ne se contente pas d'un F1 agrégé identique.

Le corpus FIT80 a déjà été exploré ; les enseignants et les modèles amont
restent historiques et ne sont pas réentraînés dans chaque pli imbriqué.
Il s'agit d'un résultat de recherche sur FIT80, pas d'un test externe indépendant.
Les autres caractéristiques et le MIDI officiel restent indépendants de cette branche.

## Reproduction

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m scripts.onset_recovery_hgb256 \
  --root DONNEES_SIGNEES --previous RESULTATS_RUN_39915038301 \
  --out NOUVEAU_DOSSIER --workers 5 --phase all
```

Le workflow `onset-score-recovery-hgb256.yml` exécute les tests à chaque push
de cette branche. Sa reproduction complète est disponible via l'entrée
`full_training_and_audio` de `workflow_dispatch`.
