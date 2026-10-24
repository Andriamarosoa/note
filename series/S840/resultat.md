---
serie: S840
statut: en_cours
type: experience
reference: S708
branche: codex/note-s840-onset-neural-latent-transposition256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S840 — Statut des tests et de l'entraînement du transpositeur neural latent

## Exécution
Prototype CNN+transport latent conditionné par demi-tons, conçu pour améliorer la détection des notes sans synthétiser de son transposé. Le chemin actuel est strictement **une nouvelle hypothèse de notes par 256 échantillons**, sans lecture future ni attente. Le code n'émet aucune attaque NoteOn et n'alimente aucun des cinq autres axes. Aucun corpus de paires de transpositions **réellement enregistrées**, certifié et disjoint au niveau des prises, n'est encore fourni dans cette série.

## Métriques
| Tête / sortie | Résultat | Signification |
|---|---:|---|
| Onset F1 de travail protégé | **69,2584331 %** | direct Acoustic195D sur 80 GuitarSet FIT80, 7 607 TP / 3 093 FP / 3 660 FN |
| Arbitre HGB rejeté | 69,2489759 % | +3 FP, 0 TP, non promu |
| Onset F1 de la transposition latente S840 | **NON MESURÉ** | aucune vraie paire ni replay 80 WAV |
| Exact-K | Inchangé / non réévalué | autre tête protégée |
| Active Exact-K | Inchangé / non réévalué | autre tête protégée |
| Offset | Inchangé / non réévalué | autre tête protégée |
| Frame Pitch F1 | Inchangé / non réévalué | autre tête protégée |
| Full Note F1 | Inchangé / non réévalué | autre tête protégée |

Un micro-F1 calculé dans le trainer sur des **masques de notes par trame** n'est pas un F1 d'événements Onset ±80 ms, et n'est jamais à comparer numériquement au 69,2584 %. Les tests synthétiques ne démontrent pas une IA qui sait déjà transposer la guitare.

## Comparaison à S708
S708/H0 à 73,252019 % est une **référence officielle protégée historique**, entraînée à une cadence 1024 et exclue du nouveau modèle. La référence de travail Onset actuelle est bien la tête directe PCM256 à 69,2584331 % sur 80 WAV. Il n'y a aucune réduction de FN mesurée ni changement d'Exact-K/Offset/Frame Pitch/Full Note sur la production ; aucune comparaison inter-architectures ne sera présentée comme score natif S840.

## Audits
| Rôle | État | Éléments |
|---|---|---|
| Données | UNKNOWN | aucune archive validée de paires de notes réellement enregistrées |
| Acoustique | PASS code | shape [N,256], pas de FFT/resampling et pas de futur |
| Architecture | PASS code | transposition par réseau appris, sortie hypothèse seule ; 5 autres axes inchangés |
| Apprentissage | UNKNOWN | backprop synthétique du logiciel, mais aucun entraînement GuitarSet réel |
| Métriques | UNKNOWN | pas de nouveau F1 Onset sur 80 WAV |
| LIVE | UNKNOWN | seulement horloge logique 256, pas de mesures p99 Windows/iRig |

## Verdict
**S840 non promu, en développement.** Il s'agit d'une implémentation fonctionnelle à tester techniquement, pas d'un meilleur score. Les conditions « transposition IA, pas de calcul de pitch shift » et « exactement 256 échantillons » sont mises en code ; leur pertinence musicale demande de véritables paires enregistrées, une ablation stricte, et une étude du cas E2 grave sans regarder plus de 256 échantillons.

## Prochaine action
Préparer une banque traçable de **vraies paires de notes / accords de guitare transposés**, sans DSP ni confusion train/cal/outer, entraîner le transpositeur, puis le comparer à l'encodeur identique **sans** vues transposées. Évaluer ensuite l'apport comme *nouvelle proposition Onset*, isolée du reste de NOTE, sur les 80 WAV signés puis sur jeu externe jamais exploré. Conserver TP/FP/FN, identifier les FN supplémentaires récupérés et FP éventuels, vérifier le respect strict de 256 et p99 CPU avant toute promotion.
