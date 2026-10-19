# NOTE S747 — Native independent specialist states, FIT80/CAL32

## Problème mesuré avant S747

S745 (12 boucles) : gain isolé d'Active Exact-K avec régressions Onset,
Offset et notes complètes. S746 a corrigé la mutation des événements
signés S708, mais sur ses 12 boucles CAL32 **aucune configuration sûre**.
S746 `depth5_risk55_tail20` : 74.3394 % Onset, 62.3084 % Exact-K,
56.6134 % Active-K, 77.2243 % Offset, 57.1138 % notes complètes,
76.6449 % Frame Pitch ; **3 anciens Offsets corrects perdus**.
Score source S708 signé : 74.2639 %, 62.0455 %, 56.2762 %,
77.3292 %, 57.1319 %, 76.4626 % dans le même ordre.

## Architecture mise en œuvre

1. S708 reste un flux de référence immutable (pitch, temps NoteOn/Off,
   reason, disponibilité), modèle et sélecteur inchangés.
2. Les 5 spécialistes acoustiques natifs supplémentaires H1–H5
   reçoivent les postérieurs, FFT/partiels et contexte causal du
   fragment courant. Chaque spécialiste possède sa propre matrice
   de période réfractaire [49 notes] ; aucun spécialiste ne modifie
   le compteur, l'Onset, l'Offset ou l'état actif d'un autre.
3. Un routeur causal H0-ancré choisit pour **chaque fragment** un masque
   parmi 32 combinaisons possibles et réunit les offres de hauteurs
   indépendantes (maximum deux offres distinctes au fragment).
4. L'activité source est un **shadow read-only** reconstitué à partir
   des seuls événements dont available_sample <= temps reçu. Aucun
   MIDI futur n'est consulté pour générer une proposition actuelle.
5. L'émetteur original S707 est exécuté **uniquement sur les notes
   supplémentaires**, jamais sur les notes signées S708. La fusion
   S746 donne priorité aux événements originaux et borne la polyphonie.
6. Deux experts HGB FIT80 sont entraînés pour estimer la probabilité
   qu'une proposition soit une instance complète correcte. Labels
   FIT pour l'entraînement, jamais dans la prédiction. Douze
   configurations (2 architectures × 3 seuils risque × 2 seuils K)
   seront vérifiées sur CAL32.

## Contraintes de promotion

Six scores calculés sur le même flux MIDI final : Onset F1,
Exact-K, Active Exact-K, Offset accuracy, Full Note F1 et Frame Pitch F1.
Si une caractéristique baisse, si une instance d'Offset
antérieurement correcte disparaît, ou si un vrai Onset manque,
aucune promotion. Le TEST réservé reste fermé. LIVE, S80,
S708/S707/S675 officiels inchangés ; pas de distillation avant 80 %.

## Validation en cours

Unit tests : préfixe temporel invariant sous mutation de données futures,
découplage des têtes, conservations d'événements S708, masque
dynamique et verrou sur les propositions de hauteurs déjà actives.
L'exécution de test sur CAL32 ne vaut **pas** preuve de gain tant que le
rapport six-métriques et les pertes d'instances ne sont pas disponibles.

- Exécution : https://github.com/Andriamarosoa/note/actions/runs/39202362375
- PR DRAFT : https://github.com/Andriamarosoa/note/pull/73
