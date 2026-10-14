# S469–S476 — Troisième source audio : MAESTRO v3, piano réel indépendant d’IDMT

**Date : 14 octobre 2026. Pas de promotion.** Cette série est un test de transfert de **V88 original**, et **PAS une reconstruction de toute la chaîne S80**. Objectif : distinguer une particularité des enregistrements IDMT d'un défaut plus général de reconnaissance polyphonique, avec une définition de K comparable à l'entraînement GuitarSet.

## Corpus et protocole de gel

MAESTRO v3 (<https://magenta.tensorflow.org/datasets/maestro>) associe WAV de piano acoustique et fichiers MIDI à une résolution d'alignement annoncée d'environ 3 ms. Les fichiers MAESTRO ne font pas partie de GuitarSet ou d'IDMT ; ce test examine volontairement **un autre instrument**, pas seulement un autre micro de guitare. Le jeu de test MAESTRO comporte une sélection officielle par composition.

- **[S469 SUCCESS #38558629021](https://github.com/Andriamarosoa/note/actions/runs/38558629021)** : lecture des noms de fichiers et durées uniquement, depuis le CSV officiel v3. Une sélection SHA reproductible de **trois compositeurs différents**, split `test`, durée 55–145 secondes, 12 premières secondes analysées commençant à +15 s. Ni WAV ni MIDI ouverts à cette étape.
- **[S470 SUCCESS #38559067956](https://github.com/Andriamarosoa/note/actions/runs/38559067956)** : téléchargements audio seuls, vrais WAV piano 44,1 kHz PCM16 stéréo depuis le miroir individuel ; conversion mono déterministe, restauration du véritable V84→V86→V87→V88 et `freeze_local_combo` historiques inchangés, et des quatre modèles S466 entraînés seulement sur GuitarSet. Toutes les **575 prédictions V88**, les probabilités et leurs SHA gelés **avant** la lecture des MIDI. V88 donne `[473,69,25,8,0,0,0]` en K0..K6, pour les 36 secondes ; S466 suggère **0** suppression, par consensus conservateur des quatre modèles.
- **[S471 SUCCESS #38559667980](https://github.com/Andriamarosoa/note/actions/runs/38559667980)** : seulement après SHA du S470, décode les vrais MIDI associés, regroupe les notes qui commencent dans une fenêtre de **20 ms**, apparie une seule fois (algorithme de couverture maximale) aux candidats V88 à **±80 ms**, et compare K avec les hauteurs qui **sonnent** encore 30 ms après l'attaque.
- **[S476 SUCCESS #38560885111](https://github.com/Andriamarosoa/note/actions/runs/38560885111)** : contrôle indispensable de la définition du K. Sur les **322 mêmes événements V88** appariés, calcule cette fois deux vérités distinctes : (a) toutes les notes encore actives, (b) les **nouvelles notes frappées** dans le groupe d'attaque, plus proche des annotations de GuitarSet. Le MIDI avait déjà été ouvert en S471 ; S476 est descriptif, non aveugle, sans retoucher les poids.

## Résultats réellement vérifiés

| Mesure, 3 extraits piano 12 s | V88 originel |
|---|---:|
| Attaques MIDI regroupées | **330** |
| Attaques retrouvées avec le gel V88 (±80 ms) | **322/330 = 97,58 %** |
| Candidats acoustiques V88 générés | **575** |
| Candidats à plus de 80 ms de tous les onsets MIDI | **64/575 = 11,13 %** |
| K0 **prédit** parmi les 575 candidats | **473/575** |
| Exact-K sur les 322 attaques — toutes notes encore actives | **24/322 = 7,45 %** |
| Exact-K poly — toutes notes encore actives | **6/232 = 2,59 %** |
| Exact-K sur les 322 attaques — **nouvelles notes frappées** | **42/322 = 13,04 %** |
| Exact-K poly — **nouvelles notes frappées** | **7/73 = 9,59 %** |

**179/322 onsets associés** ont des cardinalités MIDI différentes entre « nouvelles notes » et « notes encore actives ». Ce biais explique une partie de l'écart du premier score piano S471 ; **ne jamais présenter 2,59 % sans exposer les 9,59 % obtenus avec une vérité d'attaque plus comparable**. Les notes retenues à la pédale ne sont pas complètement modélisées par la simple durée note-on/note-off MIDI ; cette limitation doit être mentionnée.

Sur les 3 morceaux, les départs MIDI sont identifiés presque intégralement par V88, ce qui exclut un défaut d'appariement temporel général comme explication dominante. Néanmoins, beaucoup sont prédits K0 alors qu'une frappe est annotée.

## Comparaison aux autres sources : deux phénomènes distincts, et attention aux dénominateurs

- **IDMT Dataset3**, cinq WAV guitare 10 s : 47 vraies attaques regroupées, **47/47** retrouvées, **506** événements V88, **418/506 = 82,61 %** à plus de 80 ms de toute attaque et 417 K0 prédit parmi ceux-ci. La surproduction d'événements éloignés des attaques est particulièrement marquée sur 4/5 WAV, selon S462–S464.
- **MAESTRO**, trois WAV de piano 12 s : **322/330** attaques retrouvées, **575** événements, **64/575 = 11,13 %** éloignés des attaques. L’excès de candidats éloignés n'est pas de même ampleur qu'IDMT.
- **Le comptage K est néanmoins fragile sur une troisième source**, indépendamment des candidats éloignés : sur les attaques réellement reconnues, V88 ne reconnaît que **42/322** comptages de nouvelles notes, dont **7/73** accords poly. Ceci confirme un défi de transfert du modèle d'origine V88 au piano, mais ne permet pas d'attribuer toute la différence à la « source IDMT ».
- GuitarSet original V88 : **2 530/7 385** vrais accords poly sur des exemples historiques (S473), mais l’historique d’exposition à GuitarSet empêche d’en faire un benchmark aveugle.
- Les scores supérieurs de **S80 GuitarSet 81,98 % poly** et **S80 IDMT 7/21 = 33,33 % poly** sont des évaluations d'une **AUTRE chaîne**. Comparer directement ce S80 avec le V88 MAESTRO serait trompeur : **S80 complet n'est pas encore porté sur MAESTRO**.

## Origine possible des erreurs et architecture

Les fenêtres V8.6 de pics locaux (fusion initiale d'environ 4 échantillons de 44,1 kHz) suivies de la consolidation à 40 ms de V90 peuvent créer des événements supplémentaires pendant un son soutenu, d'où la surproduction IDMT. Cette cause est compatible avec le signal RMS du S464 : seulement 3/417 K0 prédits loin des attaques IDMT présentent une nouvelle montée de RMS >6 dB. **Mais MAESTRO démontre aussi que les erreurs de cardinalité surviennent sans cet excès d'événements éloignés**.

Le réseau séparé naissance/continuation S466, pourtant utile sur GuitarSet et prudent sur IDMT, n'a proposé aucune suppression sur le piano MAESTRO. Il n'est donc pas suffisamment transférable pour devenir un nouveau filtre global. Il faut un vrai expert polyphonique de composantes harmoniques fondamentales indépendantes / nombre de notes naissantes, séparé du détecteur d'onsets et avec audit individualisé des transitions K2↔K3↔K4.

## Verrous scientifiques à conserver

- Ces 3 morceaux sont **une petite cohorte exploratoire** de piano réel ; ils ne prouvent pas la précision universelle sur le piano.
- Les fichiers MIDI étaient scellés au moment de S470, mais ils sont désormais connus par S471/S476 ; ne pas y régler les poids ou seuils d'un candidat futur puis revendiquer une validation aveugle.
- Le jeu V88 mono/poly n'inclut pas les politiques S5, S8, S18, S72, S79, S80 originales ; **aucun score officiel S80 piano** disponible.
- Aucun K original modifié ; aucune suppression de candidat live ; aucune modification du détecteur onset/offset original et du décodeur des noms E3/G3/B3. **Pas de promotion.**

## Suivants : comparaison à protocole identique + spécialiste K

[S472](https://github.com/Andriamarosoa/note/actions/runs/38559972349) a montré que la seule probabilité de naissance de S466 n'est pas assez informative pour faire K2↔K3 ou K3↔K4 (AUC proche de 0,5). [S473](https://github.com/Andriamarosoa/note/actions/runs/38560454389) a donc entraîné 12 experts directionnels de morphologie pour les classes **K2, K3, K4** de la référence V88, sans toucher S80 ; aucun seuil assez sûr n'a été sélectionné sur GuitarSet (0 événement corrigé et 0 régression). Le prochain audit de risque détaillé est S474.

La prochaine validation exploitable nécessite une chaîne **S80 complète** appliquée à la même cohorte MAESTRO et/ou plusieurs sources inédites instrument-par-instrument, avec labels « nouvelles notes » fixés a priori et séparation des métriques attaque/Exact-K/noms des notes, avant toute promotion.

## S477 — pourquoi la régression dépend très fortement de la source

[**S477 SUCCESS #38561913315**](https://github.com/Andriamarosoa/note/actions/runs/38561913315) : audit **en lecture seule** de la matrice de confusion du **même V88 d'origine**, à partir des vérités S476 (piano nouvelles notes) et S453 (guitare IDMT notes actives). Les populations/annotations ne sont **pas identiques**, ne pas les fusionner comme un seul benchmark.

| Erreur au niveau des vraies attaques | MAESTRO piano, nouvelles notes | IDMT guitare, notes actives |
|---|---:|---:|
| Attaques évaluables | 322 | 45 |
| V88 Exact-K correct | 42 | 22 |
| K **sous-estimé** | **261** | 10 |
| K **surestimé** | 19 | **13** |
| Attaques réelles classées K0 | **233** | 7 |
| Vrais accords polyphoniques | 73 | 21 |
| Poly Corrects | 7 | 4 |
| Poly **sous-estimés** | **65** | 6 |
| Poly **surestimés** | 1 | **11** |
| Accords poly réels classés K0 | **37** | 3 |

**Découverte importante : le sens de l'erreur change avec les données.** Sur le piano, le V88 original fait massivement un **effondrement vers K0 / sous-comptage** (261/322 sous-estimés), y compris 37 vrais accords poly classés K0. Sur IDMT guitare, les vrais accords poly sont plutôt **surcomptés** (11/21), conformément à S453. Un correcteur global « augmenter K » ou « diminuer K » améliorerait un domaine au détriment de l'autre. Il faut un **routage de fiabilité conditionné par des preuves acoustiques**, et idéalement un détecteur qui puisse s'abstenir lorsque la source diffère du domaine entraîné.

Cette observation n'attribue pas tout à l'instrument : la convention K (nouveaux onsets piano vs notes actives XML guitare), la nature des accords et les petits effectifs IDMT diffèrent. Néanmoins, elle montre qu'**IDMT n'est pas la seule source où V88 échoue**, et que le mécanisme d'erreur n'est pas unique. Ces nombres sont de **V88**, toujours pas de la chaîne S80 complète.
