# S437–S438 — Validation diagnostique WAV réel → noms de notes avec K S80 verrouillé

**14 octobre 2026.** Objectif demandé : reconnaître directement les **noms de notes** (ex. `E3`, `[E3,G3]`), **sans créer de fichier MIDI**, en préservant Exact-K S80 et en différant le Live et les Offsets.

## Protocole : douze enregistrements de guitare authentiques

[S437 : code des prédictions](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s437_real_mic_balanced_K_note_names_12pieces.py) · [GitHub Actions #38523251921, SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38523251921), artefact `11875236011`.

L'audio provient de la véritable archive GuitarSet **`audio_mono-mic.zip`** (MD5 vérifié : `275966d6610ac34999b58426beb119c3`), et non de sinusoïdes synthétiques.

Les morceaux sont choisis par un hachage stable de leur **nom**, en exigeant **douze compositions distinctes** ; le choix ne consulte aucune note, aucune annotation ni aucun score de qualité. Les événements sont **échantillonnés selon le K déjà prédit par S80**, avec des quotas séparés K0, K1, K2, K3 et K4+ afin de ne pas masquer les erreurs derrière les K0.

Le code reconstruit les trajectoires originales **42×49** (FFT 4 096, hop 256, 6 harmoniques) à partir des véritables enregistrements MIC, puis utilise réellement les poids S402/S416/S428 dans l'inférence native S430. Les sorties `E3` / `G3` / etc. sont gelées dans un JSONL AVANT consultation des JAMS GuitarSet.

| K **prédit** par S80 | Nombre d'événements |
|---|---:|
| K0 | 24 |
| K1 | 45 |
| K2 | 45 |
| K3 | 40 |
| K4 | 15 |
| K5 | 5 |
| K6 | 0 |
| **Total** | **174** |

Le réseau a fourni **340 noms de notes**, soit la somme exacte des K prédits. **Aucune modification des décisions S80 pour ces événements.**

## S438 — Audit sur les véritables annotations, après gel

[S438 : code d'audit](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s438_audit_frozen_12real_guitars_by_true_K_no_K0_inflation.py) · [GitHub Actions #38523463170, SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38523463170), artefact `11874866530`.

L'auditeur charge uniquement les prédictions **déjà gelées** par S437, puis ouvre `annotation.zip` officiel (MD5 `b39b78e63d3446f2e54ddb7a54df9b10`). Chaque JAMS est apparié à **l'intégralité des vrais événements natifs de la composition**, avant de ne noter que les événements préalablement choisis. Les nombres de notes annotés de chaque événement correspondent à la référence historique S258. Aucun modèle n'est réentraîné, aucun seuil n'est choisi en lisant ces annotations.

| Classe **réelle** | Nombre d'événements | Accords dont tous les noms de notes sont exacts | Exact-K S80 correct |
|---|---:|---:|---:|
| K0 | 30 | 21 (70,00 %) | 21 (70,00 %) |
| K1 | 54 | 30 (55,56 %) | 40 (74,07 %) |
| K2 | 38 | 12 (31,58 %) | 30 (78,95 %) |
| K3 | 31 | 12 (38,71 %) | 26 (83,87 %) |
| K4+ | 21 | 3 (14,29 %) | 18 (85,71 %) |
| **Poly K2+** | **90** | **27 (30,00 %)** | **74 (82,22 %)** |
| **Nouvelles attaques K1+** | **144** | **57 (39,58 %)** | **114 (79,17 %)** |
| **Tous événements** | **174** | **78 (44,83 %)** | **135 (77,59 %)** |

Pour les 90 accords polyphoniques de ce diagnostic, le rappel/precision combiné F1 des notes individuelles est **70,2913 %** (**181** identités de notes communes sur **257** vraies notes et **258** notes prédites).

**Analyse des causes probables** : 74/90 accords ont leur **nombre K** correct mais seulement 27/90 ont toutes leurs hauteurs correctes. Sous K correct, la reconnaissance poly d'accords exacts n'atteint que **27/74 = 36,49 %**, et **47** accords ont donc un comptage correct avec au moins une hauteur erronée. Les 16 autres accords ne peuvent mathématiquement pas être exacts puisque leur K est incorrect.

Catégories de confusions issues de l'appariement approximatif de hauteurs (elles indiquent des symptômes, pas une preuve physique de leur origine) : **47 intervalles divers, 40 notes supplémentaires, 11 fondamentales manquantes, 11 cas quinte/harmonique 19, 9 erreurs d'octave, 5 erreurs de deux octaves, 8 notes voisines**.

## Pourquoi ces résultats ne remplacent PAS les références

Ce protocole est une **sélection artificiellement équilibrée par K prédit** sur douze morceaux GuitarSet ; le score global 44,83 % ne prétend pas être un score représentatif de la population. Les poids S80 ont une **exposition historique à GuitarSet**, tandis que les WAV MIC diffèrent des WAV `pickup_mix` utilisés pour constituer les caractéristiques S258, et S430 est un ensemble de quatre modèles dont l'efficacité sur nouveaux sons n'était pas validée de manière composition-disjointe. **Aucune preuve indépendante de généralisation n'est acquise.**

Les scores historiques restent **distincts** :
- Référence Exact-K **S80/S397** sur GuitarSet historique : 53 529/59 309 = **90,25 % global**, 6 054/7 385 = **81,98 % poly**, non indépendants.
- Meilleur challenger **S428** pour l'identité des notes sur les caractéristiques originales GuitarSet : 3 025/7 385 = **40,96 % accords poly exacts**, également développement historique.
- Meilleur challenger de notes complètes avec Offsets **S425** : **48,35 % F1** sur son protocole propre, non affecté par les expériences S433–S438.
- **S431** correction d'octave essayée puis refusée : 10 accords corrigés, 10 régressions.
- Live S424, S408 Offset, S402 référence hauteur, S80 comptage : **inchangés**.

## Priorités techniques fondées sur les erreurs S438

1. **Fondamentales et voix simultanées :** adapter la reconnaissance pour comparer des **combinaisons entières de hauteurs** plutôt que des scores individuels puis Top-K. Les 47 accords où K est correct mais les notes sont fausses constituent un sous-ensemble prioritaire.
2. **Résistance au changement de source audio :** mesurer séparément `pickup_mix` et `mic` sur les mêmes notes, avec normalisation acoustique spécifique entraînée uniquement sur d'autres compositions ; ne pas corriger les 12 pistes du test par seuils posthoc.
3. **K4 et plus :** seulement 3/21 vrais accords K4+ entièrement reconnus. Envisager une sélection structurée sous contraintes de cordes et de fondamentale réelle.
4. **Chaîne WAV autonome :** S433 sait raccorder les décisions S80 existantes et S437 vérifie plusieurs vrais WAV, mais il faut encore intégrer comme **un seul exécutable** tous les producteurs originaux V84–V88, S45 (16), S46 (193), S69–S72, S79 et S80 sur WAV inconnu. **L'Exact-K ne doit jamais être remplacé silencieusement par un simple comptage de pics harmoniques.**
5. **Validation indépendante préalable à toute promotion :** morceaux et instruments absents de l'entraînement S80 et des têtes de hauteur, en gardant les mêmes métriques K0–K6 et noms de notes pour tous les modèles comparés.
