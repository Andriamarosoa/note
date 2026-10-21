---
serie: S831
statut: en_cours
type: experience
reference: S708
branche: codex/note-s831-audit-corrections256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S831 — Corrections après audit du code S809–S830

## Objectif
Corriger les défauts constatés dans S830 **sans** partager le moindre poids, état, posterior ou décodeur avec les cinq autres caractéristiques. Conserver les 256 échantillons à 44,1 kHz (5,805 ms de cadence), sans lookahead, sans hold ajouté et sans toucher au MIDI final.

## Hypothèse
Les experts S830 manquaient de relations entre hauteurs et interprétaient la forte probabilité d'absence comme confiance dans des notes. Une représentation acoustique plus riche, des cibles d'attaques horodatées, un routeur positif et un audit de chaque candidat sur CAL et OUTER peuvent identifier des corrections réelles de polyphonie que le score agrégé masquait.

## Modifications
- `src/causal_note/s830_private_musical_evidence256.py` : porter les 73 features par note à 85 grâce au contexte harmonique -12/-19/-24/-28/+12/+19, voisins et ratios ; priorité à de vraies attaques répétées ; correction confiance `positive_note_confidence()` ; sélection locale prudente `rescue`; évaluation conditionnelle des experts, sans appel musical en mode S829 ; nouvelle instrumentation de coût CPU intégrant le prétraitement.
- `scripts/s831_train_audit_corrected256.py` : chaque piste fournit des vraies cibles Onset/Release horodatées ; toutes les métriques CAL de chaque variante, et chaque proposition OUTER avec corrections et régressions par K, enregistrées indépendamment de la sélection.
- `test/test_s831_audit_corrected256.py` : E2/E3, silence de probabilité 0,01, deux attaques au même pitch, futur perturbé après saturation FFT et mode S829 qui évite les nouvelles têtes.
- Test hérité S830 adapté aux features 85 et stockage float16 identique à FIT80. Les cinq autres axes, S708, S828 et LIVE officiel restent intouchés.

## Protocole
Comparer sur exactement les 80 pistes GuitarSet FIT80 signées S800, cinq folds disjoints 48 TRAIN/16 CAL/16 OUTER, sur les 306 536 frontières PCM256 incluant K0–K6. Réentraîner les cinq têtes privées par fold, sélectionner uniquement sur CAL, auditer les décisions de chaque autre expert sur OUTER **sans les utiliser pour décider**. Référence S829 appariée : 57,5247 % sur FIT80 ; H0 historique apparié 61,6972 %. Les métriques S809 SHADOW1024 ne sont pas directement comparables.

## Audits obligatoires
- Données : intégrité S800 FIT80, SHA audio et 80 pistes, protection CAL/HELD/TEST.
- Acoustique : FFT 256 causale après fenêtre de 8 192, permutation E2/E3 et attaques répétées, provenance labels.
- Architecture : cinq modèles propres à Exact-K, aucun poids ni prédiction inter-axes, cinq autres chemins inchangés.
- Apprentissage : trois têtes musicales et deux K appris sur TRAIN disjoint ; seuil/sélecteur calibré sur CAL.
- Métriques : détail K0–K6, chacune des propositions, corrections/régressions per-K, performances appariées.
- LIVE : test de parité off/on de 64+ hops, coûts CPU observés hors I/O ; p95/p99 matériels iRig encore à réaliser.
- Agents NVIDIA / Work : non invoqués ; aucun accord collectif revendiqué.

## Statut
En cours, aucune promotion. Un succès d'architecture ou de test unitaire n'autorise pas à annoncer une amélioration des scores sur GuitarSet ou de la latence matérielle.

## Rejeu scientifique FIT80
Run demandé après succès des tests techniques S831 : 80 pistes, cinq folds, candidats CAL et OUTER, métriques par K et veto contre S829. Les labels OUTER ne seront employés que pour l'audit post-prédiction.
