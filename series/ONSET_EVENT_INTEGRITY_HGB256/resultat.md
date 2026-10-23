# Résultat — Onset Event Integrity / HGB256

[Run 39915038301](https://github.com/Andriamarosoa/note/actions/runs/39915038301),
terminé avec succès le 23 octobre 2026. Code exécuté :
`c9c06c7f14624e48dd1a5f009cbcb7ce5fc47dae`.

Deux passages complets des 80 WAV originaux : audit apparié puis nouveau
replay après entraînement et gel des choix. 306536 blocs de 256 échantillons
par passage, 11267 attaques de référence. Les événements du replay final
reproduisent exactement les prédictions scellées, ainsi que les observations
causales du premier passage.

| Variante | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Contrôle69 historique | 8127 | 4519 | 3140 | 67,971396 % |
| Contrôle69 + correctif NoteOff | 8127 | 4527 | 3140 | 67,948664 % |
| Choix HGB/contrôle figé sur CAL | 8012 | 4264 | 3255 | **68,062694 %** |

Le gain global est de **0,091297 point** contre le meilleur contrôle
comparable. Il retire 255 fausses attaques, mais perd 115 vraies attaques.
La précision passe de 64,265380 % à 65,265559 % ; le rappel baisse de
72,131002 % à 71,110322 %. Ce gain n'est pas uniformément réparti.

| Composition OUTER | Contrôle F1 | Nouveau F1 | Delta points | Choix sur CAL |
|---|---:|---:|---:|---|
| BN1 | 67,550790 % | 68,004587 % | +0,453797 | HGB 7 feuilles, seuil 0,36 |
| Funk2 | 71,743569 % | 71,903199 % | +0,159631 | HGB 7 feuilles, seuil 0,28 |
| Jazz1 | 71,161271 % | 71,207615 % | +0,046343 | HGB 7 feuilles, seuil 0,16 |
| Rock3 | 68,435833 % | 68,426225 % | −0,009608 | Contrôle69, seuil 0 |
| SS1 | 61,742272 % | 61,312314 % | −0,429959 | HGB 7 feuilles, seuil 0,42 |

28 pistes progressent, 19 régressent et 33 restent identiques en F1.
SS1 concentre 99 des 115 TP perdus. Aucun seuil n'a été retouché après la
lecture de ces résultats OUTER.

## Audit du défaut NoteOff

L'ancien arbitre accepte 1253 NoteOff parmi 22380 propositions NoteOff.
Ceux-ci peuvent consommer un emplacement ou avancer l'état de réattaque.
Le correctif exclut les NoteOff avant toute mutation de cet état.
Sur FIT80, son effet mesuré est **0 TP et +8 FP**. Le contre-exemple unitaire
prouve le défaut, mais le correctif n'améliore pas le F1 de ce corpus.

## Contraintes et portée

Chaque décision utilise le bloc courant de 256 échantillons à 44100 Hz.
Audio futur : 0 ; attente ajoutée : 0 ; aucune rétrodatation. La tolérance
de scoring ±80 ms s'applique après l'inférence et n'est pas une attente audio.

Dix nouveaux HGB ont été entraînés avec trois compositions TRAIN et une CAL
pour chaque OUTER. Les poids historiques S613/S701 ont déjà vu une partie
de FIT80, et le corpus a été réutilisé dans plusieurs expériences : ce score
ne constitue pas une preuve de généralisation sur des enregistrements inédits.
Les cinq plis restent des checkpoints de recherche ; pas de promotion MIDI
ni de modèle unique de production. Le run ne mesure pas la latence matérielle.

Les experts amont des autres compositions sont réutilisés dans leurs plis
historiques ; ils ne sont pas réentraînés à l'intérieur de chaque nouveau pli
HGB. La séparation TRAIN/CAL/OUTER de la nouvelle tête ne constitue donc pas
une validation imbriquée indépendante de toute la chaîne.

## Preuves conservées

Rapports complets dans `evidence/`, modèles sélectionnés et dépendances dans
`model/onset256_research_39915038301/`. Archive originale du run :
artefact `12486215367`, SHA256
`da9e161b7ce16b00e71bd6f8b3e2c5d69619691d01fe4538691032ee8c5d9523`.
