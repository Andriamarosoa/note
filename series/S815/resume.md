---
serie: S815
statut: en_cours
type: experience
reference: S708
branche: codex/note-s815-activek-pitch-evidence
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S815 — Trois têtes indépendantes pour Active Exact-K

## Objectif

Dépasser la précision insuffisante de S814 sans modifier les sorties d'aucune autre caractéristique, et maintenir un véritable flux acoustique causal MIDI256. Objectif ultérieur supérieur à 85 % sur une cohorte d'évaluation commune, mais aucune promotion sur FIT80 seulement.

## Hypothèse

S814 réduit les sous-comptages (−6363) mais ajoute trop de surcomptages (+12216). Un expert d'occupation acoustique par hauteur, entraîné uniquement à l'intérieur de l'axe Active-K, peut offrir une estimation complémentaire sans importer ni remplacer la tête Frame Pitch.

## Modifications

Trois têtes à poids privés : classification directe K0–K6, régression K continue et expert probabiliste d'occupation des 49 hauteurs entraîné spécifiquement pour Active-K. Trois modèles distincts, cinq sélecteurs causaux et trois décalages pré-déclarés (-1, 0, +1), sélection après apprentissage seulement sur une composition interne de calibration. Pas de maintien temporel, d'accès à un postérieur S708, ni de poids d'une autre caractéristique.

## Protocole

Même source S800 FIT80 audio intégral signé, même extraction acoustique S813 à 256 échantillons, cinq plis par composition (48 entraînement, 16 calibration, 16 test). Chaque prédiction ne reçoit que des features acoustiques courantes et historiques. Les labels d'occupation servent uniquement à entraîner la troisième tête, jamais à son inférence. Comptes H0 MIDI vérifiés trame par trame (142213 / 249411 actives).

## Audits obligatoires

Validation structurelle, causalité mutation de suffixe, découpage invariant, absence de fuite entre compositions, classe K0–K6, exactitude sur seules trames actives, reproduction signée du H0 et non-régression des autres sorties via absence de modification historique. Audits de généralisation externe CAL/HELD/TEST et de latence physique iRig exigés avant toute éventuelle promotion. Les avis NVIDIA ne sont pas présumés réalisés.

## Statut

Implémentation expérimentale et tests écrits dans branche S815, aucun score musical vérifié à ce stade. L'intégration MIDI et le LIVE restent désactivés, la référence S708 préservée.
