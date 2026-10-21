---
serie: S829
statut: en_cours
type: experience
reference: S708
branche: codex/note-s829-s809-exactk-native256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S829 — Exact-K indépendant sur PCM256, héritage S809/S828

## Objectif
Transposer la logique de têtes, sélecteurs et décodeurs privés S809 sur l'entrée acoustique native256 validée par S828 ; dépasser d'abord le témoin sans prétendre préserver les 84,24 % SHADOW S809 à 1024. Déterminer la précision Exact-K sur toutes les trames K0..K6, y compris le silence, à cadence 256 échantillons.

## Hypothèse
Un readout K indépendant réentraîné à 256, avec choix du sélecteur sur inner-CAL exclusivement, peut exploiter la réévaluation S708 288D chaque 256 sans regarder les décisions des autres axes. Un encodeur S708 gelé est réutilisé : les poids du tronc ne sont pas indépendants, seule la nouvelle tête/son sélecteur/son état le sont.

## Modifications
Ajout isolé du module S829, du replay FIT80, des tests, d'une CI indépendante, et de ces rapports. Aucune modification de S708, S809, S810, S828, LIVE, ni du flux MIDI officiel. Aucun hold/lookahead d'inférence ajouté ; le temps CPU réel reste à mesurer. L'acoustique dépend du S708 natif original, sans postérieurs S795 obsolètes.

## Protocole
Corpus FIT80 GuitarSet original SHA256 issu de S800, cinq compositions et 80 pistes, sans ouvrir CAL32/HELD/TEST. Pour chaque composition OUTER : trois TRAIN, une inner-CAL, une OUTER ; 48/16/16 pistes au complet. Réévaluer le 288D et le K natif sur chaque PCM256. Entraîner sur TRAIN, choisir sélecteur et shift sur inner-CAL uniquement ; comparer sur OUTER aux K raw S708 et au H0 MIDI issu de la même vérité annotée. Reporter par composition, K0..K6, sous-comptages et surcomptages. S809 84,24 % / 1024 : historique non comparable ; S828 73,83 % : Active Exact-K, une autre métrique.

## Audits obligatoires
Six audits requis : Données (SHA, dénominateurs), Acoustique (extraction réellement 256, causalité), Architecture (sélecteur et mémoire privés, perturbations), Apprentissage (TRAIN/CAL/OUTER disjoints, encodage source gelé), Métriques (K0–K6 toutes trames, comparateurs identiques), LIVE (coût CPU, iRig, aucune attente supplémentaire). Trois avis NVIDIA réels requis par AGENTS.md pour validation complète, jamais simulés ; en l'absence de preuve, UNKNOWN.

## Statut
En cours. Aucun score FIT80 S829 mesuré à ce stade et aucune promotion. Référence S708 inchangée et branche S828 conservée comme témoin Active-K natif256.

Déclencheur du protocole scientifique GitHub FIT80: commit balisé [run-fit80] ; tous les scores restent NON MESURÉS jusqu'à publication du rapport scientifique vérifié.
