# S819 — Audit scientifique FINAL des politiques K4–K6 S817 sur PCM réel GuitarSet

## Provenance et contexte

Workflow original [#39553290558](https://github.com/Andriamarosoa/note/actions/runs/39553290558) **SUCCESS** ; artefact [#12324841883](https://github.com/Andriamarosoa/note/actions/runs/39553290558/artifacts/12324841883). Métriques recomptées depuis les **80 véritables pistes** GuitarSet PCM SHA, 44,1 kHz, 306536 trames à 256 échantillons et **249411 trames Active-K de vérité K>0**. Cinq paires de checkpoints privés S815/S817, signatures SHA vérifiées et scores historiques identiques avant l'étude. Le résultat du pipeline S817 précédent = 141946/249411, celui de S815 = 141946/249411, **0 décision originale modifiée**. Le workflow d'audit n'a effectué **aucun nouvel apprentissage**.

Trois vrais avis NVIDIA externes ont été reçus séparément de Gemma, Nemotron et GLM via [S819 #39552451395](https://github.com/Andriamarosoa/note/actions/runs/39552451395), accompagné de six audits **déterministes** GitHub locaux (non égaux aux six agents personnels Work). Les avis ont été contredits et hiérarchisés dans `series/S819/external_reviews_critique.md`, et la validation réelle de l'hypothèse sur la sélection donne :

## Neuf politiques précodées — comparaison a posteriori sur chaque outer fold

| Politique | Décisions modifiées | Bonnes sur 249411 | Active-K (%) | Sous-comptages | Surcomptages |
|---|---:|---:|---:|---:|---:|
| `base@0.50` (ou 0.70/0.90) | 0 | 141946 | **56,9125** | 65530 | 41935 |
| `max_high@0.90` | 1 | 141946 | **56,9125** | 65530 | 41935 |
| `max_high@0.70` | 4867 | 141263 | **56,6386** | 63893 | 44255 |
| `max_high@0.50` | 19487 | 139085 | **55,7654** | 59067 | 51259 |
| `override@0.90` | 8 | 141948 | **56,9133** | 65531 | 41932 |
| `override@0.70` | 8713 | 140924 | **56,5027** | 65779 | 42708 |
| `override@0.50` | 28865 | 138705 | **55,6130** | 63373 | 47333 |

Le micro-gain exploratoire `override@0.90` = **+2 bonnes décisions parmi 249411**, observé après le test, **ne justifie aucune promotion ni sélection rétroactive**. C'est un cas de comparaison posthoc : choisir un seuil sur ces chiffres utiliserait la vérité d'évaluation interdite pour le fold.

Parmi les changements réellement possibles à seuil 0,70, `max_high` a réduit les sous-comptages de **1637** mais augmenté les surcomptages de **2320** : déficit net **−683 bonnes décisions**. Pour seuil 0,50, les changements plus nombreux ont encore aggravé les surcomptages, malgré des sous-comptages réduits.

## Composition : pourquoi une meilleure politique locale ne suffit pas

`max_high@0.70` contre `base` change les comptes corrects : BN1 **−21**, Funk2 **+8**, Jazz1 **−134**, Rock3 **+22**, SS1 **−558**, total **−683**. Même si deux compositions obtiennent quelques gains, **les sélectionner après avoir regardé les outer scores est interdit**. Le résultat montre surtout une mauvaise transférabilité des preuves hautes polyphonies.

Porte `K>=4` activée :
- 0,50 : BN1 6996, Funk2 6141, Jazz1 4979, Rock3 13118, SS1 13934 frames acoustiques.
- 0,70 : BN1 1694, Funk2 2614, Jazz1 1670, Rock3 3070, SS1 5289.
- 0,90 : BN1 0, Funk2 12, Jazz1 0, Rock3 0, SS1 7.
- Malgré quelques franchissements à 0,90, politique originale S817 ne modifie aucune décision sur les 306536 frames, reproduction originale **exacte**.

## Verdict et réponse aux modèles NVIDIA

**Hypothèse confirmée :** S817 neutralisait effectivement ses nouveaux experts, faute de les sélectionner.

**Hypothèse réfutée :** abaisser simplement leur seuil de sélection ne permet pas de corriger le score global : cela augmente surtout les faux ajouts de notes. Le choix conservateur était justifié par les vrais chiffres.

**Non établi :** causes physiques fondamentales/harmoniques, défaut des labels, erreur de weighting, erreur du détecteur d'attaque/offset ; aucune nouvelle ablation d'architecture ni validation externe n'a encore été menée en S819.

**Statut final de S819 : REVUE ET DIAGNOSTIC TERMINÉS — AUCUN NOUVEAU MODÈLE SUPÉRIEUR.** Ne pas promouvoir ni fusionner : `promotion=false`, S708 et LIVE inchangés. Les six axes officiels ne sont pas réévalués. Pour une prochaine série, expérimenter des **preuves acoustiques réellement discriminantes** et un nouveau estimateur privé entraîné sur train-only, choisir sa règle sur inner-CAL uniquement, mesurer la confusion par vraie cardinalité et tester sur source extérieure lorsque la recherche FIT80 atteint un palier. Respecter causalité stricte MIDI256 256 samples / 44,1kHz ; la latence hardware totale n'est pas prouvée.
