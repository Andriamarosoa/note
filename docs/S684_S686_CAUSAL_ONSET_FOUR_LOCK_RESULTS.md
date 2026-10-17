# Audits S684–S686 — Onset sans régression de NOTE

**Date : 17 octobre 2026. État : RECHERCHE, aucune promotion LIVE.**

## Référence sur 12 extraits GuitarSet DEV

- F1 **NoteOn+hauteur MIDI ±80 ms** : **63,1046 %** (TP 998, FP 585, FN 582)
- **Exact-K décodé** : **54,4721 %** (26 675 / 48 970 trames)
- **S675 NoteOff ±50 ms** : **773 / 990 = 78,0808 %**, conditionnel aux NoteOn appariés
- **F1 note complète** : **48,8776 %** (hauteur + Onset + Offset)
- Les postérieurs natifs K0–K6 et les poids des têtes restent gelés.

Le K décodé est différent du score historique S80 sur ses propres événements.
Les scores DEV sont connus historiquement et ne sont pas des validations
prospectives sur nouvelle guitare Windows.

## S684 — localisation des erreurs

28 extraits véritables : 16 TRAIN tenus à l'écart (4 compositions), 12 DEV.

Sur DEV : 582 notes manquées, dont **501 avec Onset sous son seuil natif**
durant la fenêtre auditée de l'attaque (86,1 %). 242/585 faux positifs sont
à intervalle 12, 19 ou 24 demi-tons d'une note vraie contemporaine. Il
s'agit d'une **association**, pas de la preuve que tous sont produits par
des harmoniques.

L'audit des 501 attaques faibles donne 305 cas avec
`max native Onset >= .20` et 135 dont la **valeur maximale séparée**
de sustain est également >=.45 ; ces maxima ne sont pas nécessairement
simultanés. Les critères causaux de récupération sont donc plus sévères.

CI : https://github.com/Andriamarosoa/note/actions/runs/39061795431

## S685 — spécialiste de qualité des attaques natives

Appris sur 2 456 attaques TRAIN déjà émises, avec étiquettes
« bonne hauteur + attaque à ±80ms ». La meilleure suppression conservant
tous les vrais Onsets sur TRAIN tenu à l'écart retire **5 faux positifs**,
mais fait régresser K décodé **49,193 % -> 49,120 %**. Rejeté.

CI : https://github.com/Andriamarosoa/note/actions/runs/39062066517

## S686 — récupération sous preuve de nouveauté spectrale

Le nouveau Onset est proposé uniquement si :
posterior Onset natif >= seuil bas, sustain actuel >= 0,45,
et variation de fondamentale spectrale vs l'historique dépassant le seuil.
Seul le canal Onset peut être relevé. Tous les autres postérieurs,
le K source et S675 restent inchangés.

Politique sélectionnée sur 5 compositions TRAIN :
`rescue_0.30_rise_0.12`.

| Métrique | Référence DEV | S686 DEV |
|---|---:|---:|
| TP Onset | 998 | 1017 |
| FP Onset | 585 | 617 |
| F1 Onset | 63,10 % | **63,29 %** |
| Exact-K décodé | 54,47 % | **54,96 %** |
| S675 NoteOff corrects | 773/990 | **788/1009** |
| S675 NoteOff, proportion | 78,08 % | **78,10 %** |
| F1 note complète | 48,88 % | **49,04 %** |

**MAIS** sur les 4 compositions TRAIN tenues :
- +8 vrais Onsets pour +24 faux ;
- F1 Onset : 70,54 % -> **70,44 %** ;
- F1 note complète : 45,97 % -> **45,93 %** ;
- K décodé progresse légèrement.

**VETO** : le candidat n'est pas promu. Aucune commande LIVE par
défaut n'a été modifiée. Il faut tester des critères de robustesse
composition par composition et la séparation harmonique.

CI : https://github.com/Andriamarosoa/note/actions/runs/39062334569

## Conditions de promotion d'une nouvelle tête Onset

1. Onset F1 avec la bonne hauteur MIDI > référence sur compositions
   séparées, sans perte notable de rappel ;
2. Exact-K natif K0–K6 inchangé **et** Exact-K décodé non régressif ;
3. NoteOff S675 nombre absolu de bonnes fins, précision conditionnelle et
   longues notes non régressifs, sur cohortes définies ;
4. F1 notes complètes non régressif ;
5. Mesurer l'exécution causale sous PortAudio Windows, débit et débordements
   `input_queue_full` avant tout changement de la référence LIVE.

Aucune optimisation sur DEV ni Oracle de GT à l'inférence. Les statistiques
sur DEV servent de vérification rétrospective. Les expériences S687/S688
visent ces conditions mais demeurent expérimentales.
