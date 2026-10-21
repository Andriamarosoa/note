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
Deux têtes K0..K6 réentraînées séparément à 256 à partir de DSP strictement privé et non supervisé par les autres tâches, avec choix du sélecteur sur inner-CAL exclusivement, peuvent apprendre la cardinalité à partir du PCM sans aucune sortie d'un autre axe. Aucun encodeur indépendant gelé, aucun checkpoint S810/S828, aucune représentation neuronale partagée, aucun état d'un autre décodeur.

## Modifications
Ajout isolé du module S829, du replay FIT80, des tests, d'une CI indépendante, et de ces rapports. Aucune modification de S708, S809, S810, S828, LIVE, ni du flux MIDI officiel. La nouvelle classe S829AcousticFrontEnd calcule elle-même des FFT causales multirésolutions (2048/4096/8192 derniers échantillons PCM) ; elle possède ses tampons et ses états. Deux modèles HGB K sont entraînés de zéro sur les vues harmonique et temporelle de ces caractéristiques. Le sélecteur choisit exclusivement à partir de leurs prédictions ou du contrôle acoustique privé, sans état étranger. S829FittedStream émet une prédiction du modèle par PCM256, sans tampon de piste entière et sans attente ajoutée. Le coût CPU réel reste à mesurer.

## Protocole
Corpus FIT80 GuitarSet original SHA256 issu de S800, cinq compositions et 80 pistes, sans ouvrir CAL32/HELD/TEST. Pour chaque composition OUTER : trois TRAIN, une inner-CAL, une OUTER ; 48/16/16 pistes au complet. Réévaluer un DSP privé [49,72] puis son résumé [369] à chaque PCM256. Entraîner les deux têtes sur TRAIN, choisir sélecteur et shift sur inner-CAL uniquement ; comparer sur OUTER au comptage heuristique audio privé et au H0 MIDI issu de la même vérité annotée. Les événements H0 et labels sont autorisés seulement pour la notation et jamais comme entrée d'inférence. Reporter par composition, K0..K6, sous-comptages et surcomptages. S809 84,24 % / 1024 : historique non comparable ; S828 73,83 % : Active Exact-K, une autre métrique.

## Audits obligatoires
Six audits requis : Données (SHA, dénominateurs), Acoustique (extraction réellement 256, causalité), Architecture (sélecteur et mémoire privés, perturbations), Apprentissage (TRAIN/CAL/OUTER disjoints, encodage source gelé), Métriques (K0–K6 toutes trames, comparateurs identiques), LIVE (coût CPU, iRig, aucune attente supplémentaire). Trois avis NVIDIA réels requis par AGENTS.md pour validation complète, jamais simulés ; en l'absence de preuve, UNKNOWN.

## Statut
En cours. Aucun score FIT80 S829 mesuré à ce stade et aucune promotion. Référence S708 inchangée et branche S828 conservée comme témoin Active-K natif256.

Déclencheur du protocole scientifique GitHub FIT80: commit balisé [run-fit80] ; tous les scores restent NON MESURÉS jusqu'à publication du rapport scientifique vérifié.

Dernière révision du protocole : chaîne complète indépendante PCM→DSP→deux têtes entraînées→sélecteur→inférence online toutes les 256 valeurs ; lancement demandé sur cette version par [run-fit80]. Aucun score accepté sans rapport signé.
