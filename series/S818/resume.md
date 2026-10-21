---
serie: S818
statut: en_cours
type: experience
reference: S708
branche: codex/note-s818-activek-six-ordinal-heads
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S818 — Six têtes ordinales indépendantes Active Exact-K

## Objectif

Repartir sur une architecture fondamentalement différente pour l'axe Active Exact-K, qui n'atteint que 57,01953803 % dans H0 FIT80 et 56,91248582 % avec S815 et S817. L'objectif scientifique final reste >85 % de toutes les caractéristiques, sur une validation commune indépendante avec MIDI causal 256, sans regression globale.

## Hypothèse

Les anciennes têtes de classification K0–K6 et spécialistes K4–K6 n'améliorent pas les K5 et K6 sans surcompter K1. La décomposition en six seuils ordonnés permet à chaque tête indépendante d'étudier son propre événement acoustique : K>=1, K>=2, ... K>=6. L'agrégation monotone devrait fournir des décisions réellement différentes, au lieu des zéro changement S817.

## Modifications

Six classifieurs binaires indépendants entraînés, six poids et six fonctions de perte séparés, contexte spectral MIDI256 présent/passé seulement. Les probabilités de cardinalité sont projetées sur une suite monotone puis transformées en K0–K6 par sept règles déclarées : quatre seuils de franchissement et trois arrondis d'espérance. Un sélecteur privé choisit par précision Active Exact-K sur composition de calibration seulement, avec bris d'égalité conservateur. Les trois modèles S815 sont conservés **uniquement comme comparateur gelé**, et ne sont jamais appelés par l'enseignant S818 en inférence.

## Protocole

80 pistes GuitarSet FIT80 réextraites du PCM mono SHA certifié à 44100 Hz, 306536 frames à 256 samples, 5 compositions disjointes : 48 morceaux train, 16 inner calibration, 16 outer score. Rééquilibrage des classes rare K>=6 avec restitution pondérée de leur population réelle. Contrôle de parité exact H0 142213/249411 et S815 141946/249411 avant score S818. Auditer le dénominateur actif et chaque vrai K1–K6. FIT80 historiquement exploré, donc pas de certification externe.

## Audits obligatoires

Indépendance six modèles, absence de poids ou d'entrées Frame Pitch / Onset / Offset / Full Note / Exact-K, contrôle de causalité par mutation du futur, formes audio natif256, classe rare pondérée, selectors sur inner cal, signatures SHA, décisions MIDI inchangées côté production, absence de patience/hold/lookahead. Les six autres scores et la latence physique iRig restent à prouver pour toute promotion. Aucun modèle NVIDIA/personnel n'est présenté comme audité sans journal d'exécution réel.

## Statut

Nouvelle série de recherche S818 préparée, performances non mesurées à ce stade. Zéro changement LIVE, zéro promotion. La référence H0 et les enseignants précédents restent intacts.
