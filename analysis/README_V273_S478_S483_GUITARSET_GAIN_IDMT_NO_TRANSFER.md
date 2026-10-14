# S478–S483 : boucles de résolution du K0 erroné, avec preuve de transfert réel

**Statut : R&D seulement. Source V88 ≠ référence officielle S80. Aucune promotion.**
**14 octobre 2026.** Chaque tête est entraînée/calibrée exclusivement sur des annotations GuitarSet, sans revoir l'IDMT ou MAESTRO externe. La chaîne originale V88, la sortie historique S80, la détection Live et les offsets ne sont pas changés.

## S478 — ExtraTrees multiclasses conditionnel à un K0 prédit par V88

[CI SUCCESS #38565786548](https://github.com/Andriamarosoa/note/actions/runs/38565786548).

N=59 309 vrais événements GuitarSet S258, K original V88 avec 48 454/59 309 corrects, vrais accords poly 2 530/7 385. Apprentissage 4 folds séparés par composition et calibration interne par autres compositions. Réseau **sur les seuls événements prédits K0** : `y0` = aucune naissance annotée; `y1` = une naissance annotée; `y2` = deux nouvelles notes ou plus, réduit conservativement à une proposition K2. La cible y2 ne contient pas la vraie valeur de K3+ ; ne jamais prétendre à un Exact-K3 dans ce raccourci.

**Résultat :** aucun des 4 seuils calibrés ne passe les garanties, 0 correction retenue et 0 régression retenue. En revanche, **931 véritables événements polyphoniques** sont prédits K0 dans la source GuitarSet V88, justifiant un traitement dédié.

## S480 — pourquoi les quatre têtes ExtraTrees ont été rejetées

[CI SUCCESS #38566338372](https://github.com/Andriamarosoa/note/actions/runs/38566338372). Les 4 folds × 14 seuils = **56 évaluations internes** démontrent :

| Seuil multiclasses ExtraTrees | corrections cumulées en validation interne | régressions | net |
|---|---:|---:|---:|
| 0,45 | 211 | 375 | −164 |
| 0,50 | 63 | 83 | −20 |
| 0,55 | 22 | 12 | +10 |
| 0,60 | 8 | 2 | +6 |

Raisons chevauchantes des rejets : 55 essais avec gain net inférieur à 5, 55 essais sous 3 corrections pour 1 régression, 46 essais avec moins de 5 corrections, 11 essais avec plus de deux régressions. La réponse n'est pas un assouplissement de seuil ; il manque une représentation à forte précision positive.

## S481 — gain mesuré en sortie de validation GuitarSet : LightGBM conditionnel K0

[CI SUCCESS #38566831932](https://github.com/Andriamarosoa/note/actions/runs/38566831932). Remplacement de la famille d'arbres par **LightGBM 4×4 compositions + calibration interne séparée**, sans modifier les classes protégées K1–K6. 1 090 vrais descripteurs acoustiques originaux. Pondération racine carrée inverse de fréquence pour les classes y0/y1/y2+.

Tous les 4 folds ont choisi un seuil à partir du groupe de calibration GuitarSet uniquement (0,70 ; 0,75 ; 0,70 ; 0,75). Les **résultats des compositions tenues à l'écart** donnent :

| Mesure | Résultat |
|---|---:|
| Source V88 original global | 48 454 / 59 309 |
| V88 expérimental nouveau | **48 518 / 59 309** |
| Corrections réelles | **103** |
| Régressions réelles | **39** |
| Net global | **+64** |
| Source vrais accords poly corrects | 2 530 / 7 385 |
| Candidat vrais accords poly corrects | **2 537 / 7 385** |
| Corrections accords poly | **7** |
| Régressions accords poly | **0** |

Ce candidat est un progrès plus substantiel que S478, mais **non promouvable** tant que les 39 régressions subsistent et tant que le transfert externe ne fonctionne pas. Le sous-modèle `true K2plus` prédit seulement K2 ; l'essentiel des corrections concerne K0→K1, pas K3/K4.

## S482 — GEL sans ouvrir les labels IDMT

[CI SUCCESS #38567795897](https://github.com/Andriamarosoa/note/actions/runs/38567795897). Restaurer les 4 vrais modèles SHA du S481, les seuils originaux GuitarSet, les 506 événements IDMT originaux S277 et leurs vrais descripteurs acoustiques S283. Les deux règles pré-déclarées sont consensus 3/4 et 4/4 experts, sur **seuls K0 originaux** ; un événement original prédit K1–K6 reste intact.

Résultat **aucun vote de sauvetage** sous ces seuils appris sur GuitarSet. `0` proposition K0→K1, `0` K0→K2 dans les deux politiques. Exact original V88 IDMT source K-hist **[457,26,5,8,10,0,0]** ; **ne pas confondre** avec le S80 final IDMT **[456,26,11,7,6,0,0]**.

## S483 — Validation couplée indépendante post-gel

[CI SUCCESS #38568089330](https://github.com/Andriamarosoa/note/actions/runs/38568089330). Utilise les mêmes 45 attaques réelles supportées / 21 vrais accords poly dans les 5 WAV IDMT. Le K de référence est **V88 original**, pas S80 final : 22/45 corrects, 4/21 poly. Le modèle S481 consensus 3/4 et 4/4 conserve exactement ces scores, **0 correction, 0 régression**, et 0 modification parmi les 459 événements IDMT sans annotation associée.

**Verdict** : l'amélioration de S481 sur GuitarSet ne se transfère pas directement à cette source. Il serait invalide de changer les seuils sur ces XML déjà exposés puis d'appeler le résultat « validation indépendante ». Une approche multi-instrument réellement pré-entraînée ou une adaptation sur des nouveaux morceaux annotés exclusivement pour l'entraînement est requise.

## Protection du projet

Aucune régression de la référence S80, ni de `freeze_local_combo`, ni du système Live, ni des noms de notes déjà présents : ces expériences ne touchent pas la chaîne de production. La prochaine vérification d'une architecture acoustique **multi-instrument** est S485–S488 avec Spotify Basic Pitch sur MAESTRO piano puis IDMT guitare. Le résultat doit être évalué en termes de **notes réelles (E3, G3, B3)**, et pas seulement de comptage K.
