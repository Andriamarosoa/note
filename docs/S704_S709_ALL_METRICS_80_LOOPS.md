# NOTE — bilan final des boucles S704 à S709

**Objectif > 80 % partout non atteint.** Les 19 boucles sont terminées :
3 357 comparaisons configuration × checkpoint sur CAL,
sept candidats figés et sept artefacts recomptés avec succès. Aucun candidat
ne passe les six seuils sur DEV et HELD. Aucune promotion LIVE.

L'objectif demandé est strict : **chacune des six mesures doit dépasser 80 %**, sur les mêmes événements émis. Une moyenne élevée ou six meilleurs chiffres pris dans des configurations différentes ne valident pas cet objectif.

Les scores couvrent l'Onset avec hauteur correcte (tolérance ±80 ms), le nombre exact de notes ouvertes (Exact-K), Exact-K sur trames actives seulement, l'Offset conditionnel (±50 ms), le F1 des notes complètes et le F1 des hauteurs par trame.

## Résultats sur le DEV commun

12 prises GuitarSet, 1 580 notes de référence et 48 970 trames. Chaque ligne
correspond à **une seule chaîne complète**, et non à un assemblage de meilleurs scores.

| Chaîne figée | Onset F1 | Exact-K | K actif | Offset ±50 ms | Notes complètes F1 | Hauteurs par trame F1 |
|---|---:|---:|---:|---:|---:|---:|
| Natif + S675 causal | 62,73 % | 52,84 % | 42,91 % | 74,09 % | 46,10 % | 60,32 % |
| S704 | 66,60 % | 60,47 % | 54,62 % | 67,97 % | 44,88 % | 70,36 % |
| S705 | 68,29 % | 61,09 % | 54,42 % | 72,51 % | 49,10 % | 70,34 % |
| S706 | 68,38 % | 59,34 % | 52,98 % | 72,00 % | 48,87 % | 70,23 % |
| S707 / S705 | 68,29 % | 59,32 % | 53,83 % | 72,88 % | 49,35 % | 68,94 % |
| S707 / S706 | 67,92 % | 57,99 % | 52,61 % | 73,15 % | 49,31 % | 68,49 % |
| S708 | 68,75 % | 58,97 % | 54,61 % | 73,73 % | 50,31 % | 69,25 % |
| S709 | 66,86 % | 62,50 % | 57,23 % | 73,34 % | 48,62 % | 71,47 % |

S708 possède ici le meilleur Onset et le meilleur F1 de notes complètes observés
parmi les sept candidats. C'est un constat diagnostique après évaluation, pas
une sélection validée sur un nouveau test. Son Onset passe de 62,73 % à 68,75 %
par rapport à la référence comparable : les vrais positifs passent de 992 à
1 055, les faux positifs de 591 à 434. Il reste 525 notes manquées. L'Offset
conditionnel baisse légèrement de 74,09 % à 73,73 %, bien que les fins correctes
appariées passent de 729 à 772.

S709 améliore le comptage et les hauteurs sur ce DEV : Exact-K = 62,50 %,
K actif = 57,23 %, F1 des hauteurs = 71,47 %. Son Onset est inférieur à celui
de S708. CAL retient l'époque 8 : les étapes 16 et 30 n'ont pas produit un
meilleur choix selon le critère conjoint. Les modèles restent distincts.

## Contrôles HELD, par groupe

### HELD64 — S704, S705 et S707/S705

64 prises, 9 464 notes et 264 576 trames.

| Chaîne figée | Onset F1 | Exact-K | K actif | Offset ±50 ms | Notes complètes F1 | Hauteurs par trame F1 |
|---|---:|---:|---:|---:|---:|---:|
| Natif + S675 causal | 67,30 % | 50,18 % | 43,10 % | 61,17 % | 40,59 % | 68,49 % |
| S704 | 67,80 % | 59,85 % | 55,62 % | 61,14 % | 40,87 % | 78,67 % |
| S705 | 66,94 % | 59,58 % | 54,55 % | 61,92 % | 40,83 % | 79,07 % |
| S707 / S705 | 66,98 % | 58,95 % | 54,97 % | 67,49 % | 44,53 % | 78,60 % |

### HELD32 — S706, S707/S706, S708 et S709

32 prises, 4 061 notes et 132 288 trames.

| Chaîne figée | Onset F1 | Exact-K | K actif | Offset ±50 ms | Notes complètes F1 | Hauteurs par trame F1 |
|---|---:|---:|---:|---:|---:|---:|
| Natif + S675 causal | 70,27 % | 51,81 % | 44,69 % | 61,21 % | 42,26 % | 69,20 % |
| S706 | 70,82 % | 63,06 % | 58,57 % | 64,58 % | 44,91 % | 81,89 % |
| S707 / S706 | 72,41 % | 63,08 % | 59,62 % | 68,85 % | 48,94 % | 81,65 % |
| S708 | 72,92 % | 62,24 % | 58,77 % | 68,66 % | 49,15 % | 80,96 % |
| S709 | 71,50 % | 64,67 % | 60,63 % | 66,21 % | 46,49 % | 82,83 % |

Le F1 des hauteurs de S709 atteint 82,83 % sur HELD32. Les cinq autres mesures
restent sous 80 %. Ce dépassement isolé ne valide pas l'objectif global et
ne se compare pas directement au groupe HELD64.

## Boucles effectivement exécutées

| Série | Boucles | Configurations × checkpoints sur CAL | Choix CAL |
|---|---:|---:|---|
| S704 | 3 | 288 | 220 arbres |
| S705 | 3 | 288 | époque 32 |
| S706 | 3 | 288 | époque 10 |
| S707 / S705 | 3 | 870 | S705 époque 32 + S675 |
| S707 / S706 | 3 | 780 | S706 époque 20 + S675 |
| S708 | 1 | 411 | S706 époque 10 + guide K + S675 |
| S709 | 3 | 432 | époque 8, six cordes |

Il s'agit de 12 points de contrôle d'apprentissage et 7 passages d'affinage.
Les comparaisons comptent les configurations enregistrées par checkpoint ;
elles ne représentent pas 3 357 entraînements indépendants.

- **S704** : Arbres de gradient, 192 entrées par hauteur, budgets 100 / 220 / 400.
- **S705** : Réseau temporel causal et interactions harmoniques, 192 entrées, époques 10 / 20 / 32.
- **S706** : 288 entrées : ajout de spectres interpolés, contraste fréquentiel et évolution de phase ; époques 10 / 20 / 40.
- **S707 / S705** : Affinage CAL des trois checkpoints S705 : seuils d'attaque, maintien/release puis S675 causal.
- **S707 / S706** : Même affinage CAL des trois checkpoints S706.
- **S708** : Propositions faibles conditionnées par K et une montée acoustique ; contrôle sans guide et affinage des fins.
- **S709** : Six distributions de corde supervisées sur FIT, encodeur S706 réutilisé, époques 8 / 16 / 30 ; K et présence déduits des six distributions.

S708 exige une montée acoustique pour proposer une note faible : K seul ne
crée pas d'attaque. S709 apprend six distributions à partir des annotations
de corde de FIT, avec réutilisation de l'encodeur S706. À l'inférence, K et
la présence des hauteurs sont calculés à partir de ces distributions prédites :
aucune annotation de corde, aucun K réel ni signal hexaphonique ne sont fournis.

## Séparation des données et portée des résultats

Les choix de modèle et de paramètres utilisent CAL. HELD et DEV servent à évaluer les candidats figés ; aucun paramètre n'est choisi par le script à partir de leurs scores. Les groupes ont néanmoins été examinés au cours de la recherche : ils ne sont pas un test extérieur inédit. Les anciennes têtes natives et S675 ont aussi une exposition historique plus large. Les labels TEST restent fermés.

S704, S705 et S707/S705 utilisent FIT = BN1, BN2, Funk2, Jazz1 (64 prises), CAL = Jazz2 (16), HELD = Jazz3, Rock3, SS1, SS3 (64).
S706, S707/S706, S708 et S709 utilisent FIT = BN1, Funk2, Jazz1, Rock3, SS1 (80 prises), CAL = BN2, Jazz2 (32), HELD = Jazz3, SS3 (32).
Les **12 prises DEV sont communes à tous les candidats**. Les deux agrégats HELD ne doivent pas être comparés directement.

## Diagnostic et protections

La difficulté principale observée reste l'accompagnement polyphonique. Par exemple, S705 atteint 80,28 % Exact-K sur le sous-ensemble solo DEV, mais seulement 41,90 % sur l'accompagnement. Les gains partiels ne justifient pas un score global de 80 %.

Le F1 des notes complètes vaut le F1 Onset multiplié par le taux Offset conditionnel et par la fraction d'appariements dont la fin de référence est évaluable. Ainsi 80 % d'Onset et 80 % d'Offset donnent environ 64 % de notes complètes. L'objectif complet exige donc des composantes plus fortes que 80 %.

Les protections vérifient également les vraies attaques perdues, les notes longues et les fins auparavant correctes. Les gains moyens ne suffisent pas à passer ces protections. Aucun de ces résultats ne démontre une impossibilité de progresser ; ils établissent les limites mesurées de cette série de boucles.

Les sept candidats échouent aussi aux protections complètes de non-régression
sur les deux groupes évalués. S709 perd notamment une attaque DEV de note longue
précédemment reconnue ; S708 en perd zéro sur DEV, mais une sur HELD32.
Les listes de gains/pertes appariés et les compteurs figurent dans les résumés
JSON. La version officielle reste inchangée.

## Causalité et limites de mesure

Les entrées proviennent uniquement du PCM déjà reçu. Les décisions sont émises à la réception de blocs de quatre hops, soit 1 024 échantillons / 23,22 ms à 44,1 kHz. Aucune attaque n'est antidatée et aucun détecteur MIDI extérieur n'est substitué au système natif. Les six mesures viennent du même flux sérialisé, avec au plus six notes ouvertes pour chaque candidat. La référence historique native + S675 peut conserver plus de six intervalles à cause des fins différées (jusqu'à huit sur DEV et dix sur HELD64) ; le compteur K du protocole est borné à six. Le plafond des candidats est vérifié sur leurs événements bruts, pas seulement sur ce compteur.

S675 est évalué avec franchissement causal, délais de secours connus et fermeture au moment disponible. Le 78,08 % Offset historique d'une sélection de fins séparée n'est donc pas une référence comparable ; le contrôle intégré DEV vaut 74,09 %. L'Onset historique à un hop (63,10 %) n'est pas non plus la référence à quatre hops (62,73 %). Le 66,66 % Exact-K du spécialiste S699 évalué séparément n'est pas réutilisé pour ces chaînes.

Un défaut d'ordre des événements à la fin d'un flux exactement aligné sur un bloc a été corrigé au commit a9d3bbb7099396e2d9f1cbeffb57ed528fc22e80 : les fermetures EOF passent après les événements ordinaires du dernier bloc. Un test de préfixes tronqués couvre ce cas. S707/S708 ont tourné à leurs commits antérieurs ; les prises effectivement évaluées ne se terminent pas sur cette frontière, ce que le recomptage consigne. S709 et le code final incluent le correctif.

Il s'agit de relectures causales de WAV. Les horodatages du benchmark n'incluent pas la latence du périphérique ni le temps CPU total. Le chronométrage S705 des seules têtes incrémentales (médiane 2,87 ms, p95 3,89 ms sur le runner) exclut le frontend et ne prouve pas à lui seul les performances microphone.

## Preuves et reproduction

Le workflow de recomptage télécharge les sept artefacts depuis des exécutions et commits identifiés, vérifie leur taille et leur SHA-256, puis recalcule TP/FP/FN, appariements de fins, intervalles de hauteur et Exact-K depuis les JSONL. Il ne désérialise aucun modèle. Les résultats vérifiés, les choix figés et leurs empreintes sont conservés sous analysis/evidence/s704-s709-loop-audit/.

Les ZIP d'expériences contiennent les checkpoints, les scores CAL par configuration et les événements exacts. Leur rétention GitHub est de 60 jours ; les résumés vérifiés et le code de reproduction sont versionnés dans le dépôt.

Les sept recomptages coïncident avec les rapports d'expérience. Aucun des
fichiers HELD/DEV de cette série ne se termine exactement sur une frontière
de bloc de 1 024 échantillons. Les fermetures EOF ne reçoivent aucun crédit
artificiel pour l'Offset.

Les tests S709, temporels et métriques lancés avant son apprentissage passent
(12 tests dans cette exécution). Les tests séparés de frontière EOF, de
préfixes et de sélection K passent également dans
[l'exécution 39094979574](https://github.com/Andriamarosoa/note/actions/runs/39094979574).

| Série | Exécution GitHub | Commit d'expérience | Résumé recompté |
|---|---|---|---|
| S704 | [39092626757](https://github.com/Andriamarosoa/note/actions/runs/39092626757) | [a137ae0b4f62](https://github.com/Andriamarosoa/note/commit/a137ae0b4f62141c2dd3cfdce46cd6e30a314f99) | [JSON](../analysis/evidence/s704-s709-loop-audit/s704_verified_summary.json) |
| S705 | [39092939353](https://github.com/Andriamarosoa/note/actions/runs/39092939353) | [8180a0c8847f](https://github.com/Andriamarosoa/note/commit/8180a0c8847fc598d5fb859434658d3d11ab3835) | [JSON](../analysis/evidence/s704-s709-loop-audit/s705_verified_summary.json) |
| S706 | [39093239181](https://github.com/Andriamarosoa/note/actions/runs/39093239181) | [20f8536b33d3](https://github.com/Andriamarosoa/note/commit/20f8536b33d3a839accc7d8cad0cba2dde6c445c) | [JSON](../analysis/evidence/s704-s709-loop-audit/s706_verified_summary.json) |
| S707 / S705 | [39093963095](https://github.com/Andriamarosoa/note/actions/runs/39093963095) | [2cdff0539b0f](https://github.com/Andriamarosoa/note/commit/2cdff0539b0fcbe7dfde9c02e2e9450635856643) | [JSON](../analysis/evidence/s704-s709-loop-audit/s707_705_verified_summary.json) |
| S707 / S706 | [39093963095](https://github.com/Andriamarosoa/note/actions/runs/39093963095) | [2cdff0539b0f](https://github.com/Andriamarosoa/note/commit/2cdff0539b0fcbe7dfde9c02e2e9450635856643) | [JSON](../analysis/evidence/s704-s709-loop-audit/s707_706_verified_summary.json) |
| S708 | [39094682419](https://github.com/Andriamarosoa/note/actions/runs/39094682419) | [d19f1741de1e](https://github.com/Andriamarosoa/note/commit/d19f1741de1e9cdc6e52a6f665b71a18ca36b873) | [JSON](../analysis/evidence/s704-s709-loop-audit/s708_verified_summary.json) |
| S709 | [39095466721](https://github.com/Andriamarosoa/note/actions/runs/39095466721) | [d133288208c3](https://github.com/Andriamarosoa/note/commit/d133288208c3a5220c05b0fba0ebb76222287a35) | [JSON](../analysis/evidence/s704-s709-loop-audit/s709_verified_summary.json) |

Le [recomptage final 39096563672](https://github.com/Andriamarosoa/note/actions/runs/39096563672)
s'exécute au commit `7ec8b6f1e3315159b6c1921d96106045a848addd`.
Son artefact `S704_S709_INDEPENDENT_FINAL_RECOUNT` rassemble les archives,
rapports et résultats vérifiés. Son SHA-256 est
`1de000ee9edd29ee7e1ecca7ee8517502d3ef28886df1609b194e09f2c90ae33`.

```bash
python scripts/s704_verify_emitted_artifact.py resultats.zip --out verification.json
PYTHONPATH=.:src python -m scripts.s704_s709_final_recount --out /tmp/note_recount
PYTHONPATH=.:src python -m unittest test.test_s709_string_heads test.test_s705_temporal_heads test.test_s704_joint_loops
```

Le recomptage global demande un accès GitHub en lecture via `gh`, ainsi que
NumPy et SciPy ; il ne charge aucun checkpoint. Les dépendances et commandes
d'apprentissage sont épinglées dans les workflows de chaque expérience.
