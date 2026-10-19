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

# S789 — Résultats

## Exécution

- GitHub Actions : en attente.
- SHA source : en attente.
- Corpus/split : GuitarSet CAL32, DEV 12 morceaux déjà exposés, HELD 32 morceaux déjà exposés ; S781 FIT80 et S782 CAL32.
- Changement du décodeur LIVE : aucun.

## Métriques

Les six métriques de référence signée S708 (CAL32), à **auditer** :

| Mesure | Référence CAL32 S708 | Nouveau candidat |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

## Comparaison à S708

Cette série est un **audit des données existantes**, pas une nouvelle inférence. Aucun gain ni régression de modèle revendiqué. À noter : le précédent jeu HELD a un Frame Pitch F1 supérieur à 80 %, mais HELD est exposé et sa distribution diffère de CAL.

## Audits

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | CI S789 en attente |
| Acoustique | UNKNOWN | CI S789 en attente |
| Architecture | UNKNOWN | CI S789 en attente |
| Apprentissage | UNKNOWN | CI S789 en attente |
| Métriques | UNKNOWN | CI S789 en attente |
| LIVE | UNKNOWN | CI S789 en attente |

## Verdict

Non terminé. Risques à explorer : distribution décalée des compositions, faible rappel des cordes graves, surcomptage harmonique, offsets conditionnels, mauvaise assimilation des scores shadow à une sortie MIDI.

## Prochaine action

Exécuter les six audits GitHub sur les artefacts figés, produire le consensus, puis compléter ce document avec constats chiffrés, liens et SHA. Ne pas promouvoir un candidat.
