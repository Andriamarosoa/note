---
serie: S818
statut: termine
type: experience
reference: S708
branche: codex/note-s818-activek-six-ordinal-heads
run_id: 39546770781
commit_sha: 50430c40bad0a37b990f5943d3e55a71b15f7b38
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


## Clôture scientifique vérifiée — S818

Run GitHub Actions #39546770781 : https://github.com/Andriamarosoa/note/actions/runs/39546770781 — SUCCESS, 5 plis × 6 têtes ordinales privées entraînées, extraction MIDI256 de 80 pistes GuitarSet à 44100 Hz. Artefact des 30 checkpoints, décisions et résultats #12322741581 : https://github.com/Andriamarosoa/note/actions/runs/39546770781/artifacts/12322741581.

* Vrai S818 Active-K FIT80 : **135694/249411 = 54,40578002 %**, 73512 sous-comptages et 40205 surcomptages.
* S815 figé : **141946/249411 = 56,91248582 %** ; delta S818 **−2,50670580 points**.
* H0 FIT80 historique : **142213/249411 = 57,01953803 %** ; delta S818 **−2,61375801 points**.
* Par composition : BN1 27669/47770 = 57,9213 % ; Funk2 27246/48827 = 55,8011 % ; Jazz1 22959/40548 = 56,6218 % ; Rock3 28487/54917 = 51,8728 % ; SS1 29333/57349 = 51,1482 %.

**REJET SCIENTIFIQUE :** S818 a régressé malgré des poids réellement indépendants. H0, S708, le MIDI LIVE et les cinq autres caractéristiques ne sont pas modifiés. Promotion=false, unification=false, objectif 85 % non atteint, validation externe et latence matérielle non mesurées. FIT80 est une cohorte historiquement explorée, non un benchmark indépendant certifiant la généralisation. Un second run #39546788233 a été déclenché par un correctif mineur des chemins de compilation, à contrôler séparément.
