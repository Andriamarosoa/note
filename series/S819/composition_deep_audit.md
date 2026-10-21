# S819 — Audit approfondi des écarts par composition (FIT80, vrai MIDI256)

**Statut : audit descriptif de sous-groupes terminé, contrôles CI PASS ; cause acoustique physique pas démontrée.**

## Provenance et reproductibilité

- Test de neuf politiques sur les 80 pistes originales : [GitHub run #39553290558](https://github.com/Andriamarosoa/note/actions/runs/39553290558), artefact #12324841883 (`gate_audit/report.json`).
- Annotations par cardinalité des mêmes 80 morceaux, modèles S817 et signature de provenance : [run #39543749810](https://github.com/Andriamarosoa/note/actions/runs/39543749810), artefact #12321341775 (`result/report.json`).
- Rejointures exactes index/nom/composition, SHA du rapport S817 et 80 pistes, contrôles croisés du dénominateur K>0, des changements par politique et du principe Δcorrect = −Δsurcomptage−Δsous-comptage par composition.
- Audit reproductible exécuté en CI [run #39558449623](https://github.com/Andriamarosoa/note/actions/runs/39558449623) **SUCCESS**, rapport JSON complet [artefact #12326556287](https://github.com/Andriamarosoa/note/actions/runs/39558449623/artifacts/12326556287). Script `scripts/s819_composition_subgroup_audit.py`.
- Politique **comparaison posthoc seulement** : `max_high@0.70` sur experts S817 gelés, contraste contre baseline S815. FIT80 a été largement exploré ; ne pas utiliser les résultats des compositions test pour construire/choisir une politique.

## Bilan exhaustif à seuil 0.70 — toutes les compositions

| Composition | Décisions audio modifiées | Δ bonnes décisions Active-K | Δ surcomptages | Δ sous-comptages | GT K>=4 parmi frames actives |
|---|---:|---:|---:|---:|---:|
| BN1 | 895 | −21 | +332 | −311 | 8020/47770 = 16,79 % |
| Funk2 | 1048 | +8 | +418 | −426 | 7997/48827 = 16,38 % |
| Jazz1 | 515 | −134 | +275 | −141 | 6946/40548 = 17,13 % |
| Rock3 | 626 | +22 | +247 | −269 | 16067/54917 = 29,26 % |
| SS1 | 1783 | **−558** | **+1048** | −490 | 17874/57349 = **31,17 %** |
| **Total** | **4867** | **−683** | **+2320** | **−1637** | — |

La politique réduit le nombre de frames sous-comptées, mais ajoute plus de surcomptages qu'elle n'en corrige. La précision Active-K globale régresse de **56,9125 % à 56,6386 %**. L'écart positif de Funk2 et Rock3 est anecdotique et posthoc.

## Audit par enregistrement/sous-groupe identifié

Chaque composition comporte 16 pistes. Les deux groupes ci-dessous en contiennent huit chacun.

| Composition | Groupe A : Δ correct | Groupe B : Δ correct |
|---|---|---|
| BN1 | `BN1-129-Eb` **−136** | `BN1-147-Gb` +115 |
| Funk2 | `Funk2-108-Eb` **−90** | `Funk2-119-G` +98 |
| Jazz1 | `Jazz1-130-D` **−171** | `Jazz1-200-B` +37 |
| Rock3 | `Rock3-117-Bb` −22 | `Rock3-148-C` +44 |
| SS1 | `SS1-100-C#` **−487** | `SS1-68-E` −71 |

**Sur SS1, le groupe 100-C# contient 487/558 = 87,3 % des pertes nettes.** Les 8 pistes de type `_comp` concentrent 505/558 = 90,5 % de la perte SS1 ; les 8 `_solo` perdent 53. La pire piste individuelle `01_SS1-100-C#_comp` perd **290 décisions correctes** parmi 408 changements de prédiction ; ceci est un constat d'erreur et non une attribution causale à la note C#.

Sur Rock3, les huit pistes `_comp` gagnent 61 bonnes décisions mais les `_solo` en perdent 39, produisant le bilan +22. Sur Funk2, les `_comp` gagnent 9 et les `_solo` en perdent 1. Ces gains de faible ampleur ne sont pas uniformes à travers la composition.

## Interprétation scientifique limitée

1. **Prouvé** : l'effet est très hétérogène selon la composition et les enregistrements, avec une perte SS1 dominée par le groupe `SS1-100-C#`, principalement `_comp`. La règle de cardinalité haute `max_high` ajoute trop de notes actives pour SS1.
2. **Prouvé** : SS1 contient une fraction GT K>=4 plus élevée que Funk2 ; Rock3 possède néanmoins une fraction comparable à SS1 et obtient un léger gain. Le **taux de polyphonie seul est insuffisant** pour expliquer l'écart.
3. **Non établi** : quels vrais K (K1–K6) subissent les changements **du sélecteur contrefactuel**, si les gains/losses dépendent de la fondamentale, du timbre, de l'accordage, du jeu ou de l'annotation. Le pourcentage K>=4 ici est issu des **annotations originales de S817**, non d'une matrice d'erreur par vrai K de `max_high@0.70`. L'audit par groupe n'effectue aucune écoute ou décomposition des partiels.
4. **Pas de sélection autorisée** : choisir `max_high@0.70` pour Funk2/Rock3 en observant ce tableau, tout en laissant S815 sur SS1, serait une fuite des résultats des compositions d'évaluation. Une politique conditionnelle devra apprendre sa porte en train, être choisie en inner-CAL seulement et testée dans un **nouveau** protocole tenu à l'écart de FIT80 exploré.
5. **Protection** : aucune modification à S708, aux cinq autres enseignants, au LIVE ou à l'horloge MIDI256. Ce rapport n'est **pas** une mesure de latence end-to-end physique ni un score sur source externe. Aucun candidat à promouvoir.

**Prochaine expérience de diagnostic nécessaire avant d'affirmer la cause** : calculer les matrices de transition de prédictions `base → max_high` **par vrai K et par piste**, inspecter l'évidence de fondamentale/harmoniques sur les mêmes événements et tester une correction pré-déclarée avec validation externe, sans lookahead, hold ni accès au test pour la calibration.
