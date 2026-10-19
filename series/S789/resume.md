---
serie: S789
statut: termine
type: experience
reference: S708
branche: codex/note-s789-deep-data-audit
run_id: 39274713682
commit_sha: e0e5bccb9386d7704f771d18a9a66816e4014ff4
promotion: false
---

# S789 — Résumé

## Objectif

Auditer profondément les données S708/S781/S782, à l'échelle de chaque enregistrement puis des groupes par composition, texture, registre, environnement harmonique et polyphonie. Déterminer quelle caractéristique pourrait franchir 80 % en premier **sans confondre proximité au seuil et généralisation réelle**.

## Hypothèse

L'Offset est le plus proche de 80 % sur CAL32, mais sa mesure conditionnelle aux attaques correctes et ses fortes régressions d'un morceau à l'autre rendent sa promotion incertaine. Frame Pitch présente un meilleur potentiel observable au-delà de CAL32. Les mauvais scores des graves et de K5/K6 sont liés à des défauts structurels et à un changement de distribution FIT/CAL.

## Modifications

- `scripts/s789_deep_data_microscope.py` : six audits réellement exécutés en parallèle et contrôles numériques indépendants sur les artefacts originaux.
- `test/test_s789_deep_data_microscope.py` : tests des formules métriques, intervalles descriptifs, identités et six rôles.
- `.github/workflows/note-s789-deep-data-audit.yml` : téléchargement immuable S708/S781/S782, exécution matricielle de six audits et consensus bloquant.
- `series/S789/resume.md`, `series/S789/resultat.md` : rapports obligatoires et preuve de la décision.

Aucun changement du modèle de transcription, du pipeline LIVE, de l'Offset ni de l'architecture de sélection en production.

## Protocole

Sources figées : [S708 run 39094682419](https://github.com/Andriamarosoa/note/actions/runs/39094682419), [S781 run 39256936602](https://github.com/Andriamarosoa/note/actions/runs/39256936602), [S782 run 39257378677](https://github.com/Andriamarosoa/note/actions/runs/39257378677). Audits indépendants sur DEV (12 morceaux, déjà exposé), HELD (32 morceaux, déjà exposé), CAL32 agrégé et diagnostics FIT80/CAL32. Aucun entraînement, réglage sur CAL ou ouverture de TEST.

L'audit de données recalcule les métriques à partir des TP/FP/FN et des dénominateurs, contrôle la cohérence micro/macro et les pistes individuelles. Le bootstrap sur morceaux est uniquement descriptif, car les morceaux d'une même composition sont corrélés.

## Audits obligatoires

- Données : noms/splits uniques, comptages par piste, textures/compositions, portée des échantillons et variabilité.
- Acoustique : grave E2–B2, voisinages harmoniques, superposition K1–K6, diagnostic post-attaque sans prétention causale.
- Architecture : reconstitution des confusions K5/K6, différences tête isolée 1024 vs MIDI officiel 256.
- Apprentissage : sélection 5 compositions FIT sans CAL, écart FIT/CAL des classes rares.
- Métriques : recalcule six scores, supports, nombres minimaux de corrections et variations DEV/CAL/HELD.
- LIVE : S708/S707 immuables, aucun modèle/promotions, aucune latence matérielle déduite de ces fichiers.

## Statut

**Terminé : six audits PASS, consensus PASS**, <https://github.com/Andriamarosoa/note/actions/runs/39274713682>, SHA source `e0e5bccb9386d7704f771d18a9a66816e4014ff4`. Aucune amélioration d'un modèle, aucune promotion ; résultats détaillés dans `resultat.md`.
