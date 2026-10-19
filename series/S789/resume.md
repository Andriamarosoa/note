---
serie: S789
statut: en_cours
type: experience
reference: S708
branche: codex/note-s789-deep-data-audit
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S789 — Résumé

## Objectif

Auditer les données S708/S781/S782 **au niveau des enregistrements, textures, compositions, hauteurs et cardinalités**, avec six vérifications indépendantes, et classer la probabilité raisonnable que chaque métrique atteigne 80 %.

## Hypothèse

La distance au seuil 80 % ne suffit pas à classer les métriques. Les écarts entre CAL32, DEV exposé et HELD exposé, les erreurs en accords, les registres E2–B2, les voisinages harmoniques et l'évolution K5/K6 permettent de distinguer un gain structurel d'un réglage insuffisant.

## Modifications

- Six audits indépendants sur des artefacts figés des runs S708, S781 et S782.
- Recalcul par piste et par texture des TP/FP/FN, de la précision, du rappel, des métriques de hauteur et des offsets.
- Mesure de l'hétérogénéité par morceau/composition et intervalle descriptif par rééchantillonnage de morceaux.
- Recalcul des confusions K5/K6 et du décalage FIT/CAL.
- Vérification de la causalité des preuves : S782 observe jusqu'à trois fenêtres **après** l'attaque uniquement pour les diagnostics rétrospectifs.
- Fichier `resultat.md` à compléter d'après la CI, sans annoncer de gains de modèle.

## Protocole

Les artefacts originaux sont téléchargés via GitHub Actions. Tous les rapports sont immuables et identifiés par run : S708 `39094682419`, S781 `39256936602`, S782 `39257378677`. Aucun apprentissage, changement de tête, tuning CAL, opening TEST ou promotion.

Les six contrôles calculent séparément leurs conclusions ; la synthèse est `scripts/s787_audit_consensus.py`. La différence de métrique entre CAL32 et HELD doit être explicitement indiquée avec la réserve que HELD est **déjà exposé**.

## Audits obligatoires

- Données : profils par morceau, composition, texture, couverture, incohérences.
- Acoustique : registre, superposition harmonique, attaques manquées, fenêtre causale.
- Architecture : confusions K5/K6, erreurs de décodage et horloges distinctes.
- Apprentissage : composition held-out FIT, distribution shift, contamination CAL.
- Métriques : six scores signés, supports, gaps vers 80, écarts DEV/HELD/CAL.
- LIVE : absence de mise à jour des fichiers LIVE ou modèles signés, pas de verdict temps réel depuis un audit rétrospectif.

## Statut

En cours. Nouvelle mesure scientifique : **non mesuré**. La série ne crée aucun nouveau candidat.
