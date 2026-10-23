# Résultat — audit des pertes Onset en accompagnement / polyphonie PCM256

23 octobre 2026. Branche `codex/note-onset-polyphony-loss-audit256`.

**Le mode acoustique seul, choisi sur CAL pour Rock3 et SS1, concentre 821 des 825 anciens TP perdus. Les accompagnements concentrent 802 pertes (97,21 %).** Les scores acoustiques et les choix du décodeur expliquent ces refus ; la cause physique d'une faible confiance acoustique reste ouverte.

**Reproduction complète.** Les 80 WAV signés, 306536 blocs de 256 à 44100 Hz, ont été relus. Chaque bloc conserve la parité exacte des 195 caractéristiques, des événements de référence et des événements candidats. Le 69,206204 % annonce est desormais reproduit. Tests : 96 passes. Les tests CI sont distincts du replay audio exécuté localement.

| Variante | TP | FP | FN | F1 | Precision | Rappel |
|---|---:|---:|---:|---:|---:|---:|
| HGB de référence | 8050 | 4231 | 3217 | 68,3710 % | 65,5484 % | 71,4476 % |
| Acoustique directe 195D | 7585 | 3068 | 3682 | 69,2062 % | 71,2006 % | 67,3205 % |

**Comptabilité des annotations.** 7225 attaques sont conservées, 825 anciennes attaques sont perdues, 360 attaques sont nouvellement récupérées et 2857 restent manquees dans les deux versions. Ainsi `825 - 360 = 465` FN nets supplémentaires. Les comptes portent sur des identifiants de vérité terrain stables apres appariement même hauteur / un-à-un / ±80 ms. Parmi les pertes, 27 concernent un evenement present dans la tolérance mais attribue à une autre annotation ; elles ne prouvent pas une absence acoustique.

| Texture | F1 avant | F1 apres | Rappel avant | Rappel apres | Anciens TP perdus | Nouvelles récupérations |
|---|---:|---:|---:|---:|---:|---:|
| Accompagnement | 64,8918 % | 65,3420 % | 67,6971 % | 61,5505 % | 802 | 293 |
| Solo | 77,9585 % | 78,7466 % | 81,8486 % | 83,3222 % | 23 | 67 |

**Mécanismes du décodeur, sur les 825 pertes.**

| Mécanisme observe | Pertes | Part |
|---|---:|---:|
| Score direct sous le seuil, sans passage par la référence | 724 | 87,76 % |
| Score suffisant, réarmement absent | 73 | 8,85 % |
| Compétition dans appariement un-à-un | 27 | 3,27 % |
| Exclusion réfractaire | 1 | 0,12 % |

**Concentration par composition.**

| Composition | Anciens TP perdus | Nouvelles récupérations | Delta TP net | F1 avant | F1 apres |
|---|---:|---:|---:|---:|---:|
| BN1 | 0 | 30 | +30 | 68,9205 % | 69,8511 % |
| Funk2 | 4 | 34 | +30 | 72,1603 % | 73,0074 % |
| Jazz1 | 0 | 18 | +18 | 71,7723 % | 72,2352 % |
| Rock3 | 641 | 159 | -482 | 68,3445 % | 66,1832 % |
| SS1 | 180 | 119 | -61 | 61,5218 % | 65,9884 % |

Sur Rock3, la calibration SS1 choisit `direct_only`, seuil 0,90, reset 0,50 et exclusion 14 blocs. Les 641 pertes et 159 récupérations donnent -482 TP nets. 633 pertes sont en accompagnement. Parmi ces 641 cas, 584 n'atteignent pas le seuil direct dans la tolérance, 41 traversent le seuil sans réarmement et 16 relevent de l'appariement. La médiane des maxima de score des pertes est 0,8172.

Le contre-factuel qui rétablit le passage HGB sur Rock3 récupère 629 des anciennes pertes ; il perd aussi 18 TP du candidat. Son résultat est TP 2576 / FP 1559 / FN 743 / F1 69,1173 %, contre TP 1962 / FP 648 / FN 1357 / F1 66,1832 % pour le candidat Rock3. Le coût est +911 FP. Ce test prouve l'effet du routage ; il ne constitue pas une politique sélectionnée indépendamment pour déploiement.

**Contre-experiences fixes, sur toutes les compositions.** Toutes rejouent causalement les mêmes probabilités scellées, déjà vérifiées contre l'audio, sans nouvel apprentissage ni recherche de seuil OUTER.

| Modification diagnostique | TP | FP | FN | F1 | Anciennes pertes récupérées | TP du candidat perdus |
|---|---:|---:|---:|---:|---:|---:|
| Rétablir passage HGB partout | 8376 | 4578 | 2891 | 69,1631 % | 809 | 21 |
| Exclusion réfractaire reduite à 1 bloc | 7608 | 4387 | 3659 | 65,4114 % | 4 | 0 |
| Retirer besoin de réarmement | 7688 | 4249 | 3579 | 66,2644 % | 33 | 0 |
| Union exacte de tous les événements HGB | 8478 | 9156 | 2789 | 58,6693 % | 825 | 0 |

Le passage HGB rétabli partout récupère 809 anciennes pertes, mais ajoute 1510 FP nets et donne 69,163123 %, légèrement sous le candidat. Retirer les protections de réarmement ou de doublons dégrade davantage le F1. Aucune de ces quatre politiques n'est promue.

**La texture et le nombre de notes doivent rester separes dans l'audit.** Dans la référence, le rappel à K1 annote est de 65,56 % en accompagnement (634/967), contre 82,43 % en solo (1773/2151). Sur les accompagnements K3+, il est de 67,60 % (3849/5694). Le nombre de hauteurs actives seul ne suffit donc pas à expliquer l'ecart de texture. Ces associations ne prouvent ni masquage, ni fondamentale absente, ni erreur d'annotation.

**Cible de la prochaine correction.** Apprendre un arbitrage causal par fragment entre les propositions HGB et les propositions acoustiques, en conservant les vrais événements déjà reconnus. Toute correction devra annoncer et mesurer les TP conservés, TP recuperes et FP ajoutes sur des partitions de calibration puis un OUTER scellé. Les cas acoustiques faibles, le registre et les reattaques sont des analyses secondaires à confronter à cette preuve, sans baisse globale de seuil.

Les 2857 FN persistants représentent encore 77,59 % des FN actuels. Un oracle qui recupererait les 825 seules pertes sans ajouter de FP ni perdre aucun TP atteindrait environ 73,95 % F1 ; ce calcul de diagnostic n'est pas un score de modèle. La réparation de cette regression seule ne suffit donc pas à garantir 80 %.

**Preuves et limites.** L'archive experimentale contient les dix checkpoints appris, les politiques CAL figées, les probabilités OUTER des 80 pistes, les événements scellés, les rapports du replay, les 11267 annotations auditees, le CSV des 825 pertes et les empreintes SHA256. Le code est sur cette branche ; les codes du depot ne sont pas recopiés dans l'archive. FIT80 et les enseignants historiques ont déjà été exposés à la recherche. Ce résultat n'est pas une validation externe inédite. Les sorties restent privees à Onset ; les autres axes et le MIDI officiel ne sont pas promus. Le pas de 256 ne certifie pas une latence matérielle de 5,805 ms.

Code d'audit : `5458458ed91a192870ed0946c98e27a9d2a6b477`. Code d'apprentissage et de calibration : `2e2274e9171bb27b358f450ab5842e5868a84545`.

Archive : `NOTE_ONSET_AUDIT_PCM256_2026-10-23.zip` ; SHA256 `495dbc86dda288dd22c859f80d8764f98f31c857ae2d6c745f649004578df68f`.

[CI — 96 tests réussis](https://github.com/Andriamarosoa/note/actions/runs/39945139684). Le replay audio complet et les contre-expériences sont les exécutions locales archivées, distinctes de cette CI.
