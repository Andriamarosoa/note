---
serie: S790
statut: termine
type: experience
reference: S708
branche: codex/note-s790-independent-framepitch-causal-expert
run_id: 39277971956
commit_sha: a56362661f11df109e4cfafdecb07bb38e27a7cc
promotion: false
---

# S790 — Résultats

## Exécution

- Run scientifique et six audits indépendants : [GitHub Actions #39277971956](https://github.com/Andriamarosoa/note/actions/runs/39277971956) : **SUCCESS**.
- SHA de l'expérience : `a56362661f11df109e4cfafdecb07bb38e27a7cc`.
- 80 enregistrements FIT, 32 CAL, cinq folds par composition ; HELD/TEST fermés.
- Artefacts `S790_FIT80_CAL32_FRAMEPITCH_INDEPENDENT_SHADOW` et `S790_SIX_AUDITS_CONSENSUS`.
- Référence S708 signée et S707 conservées, aucun changement de source MIDI final.

## Métriques

Les valeurs officielles sont **gelées** : aucune nouvelle métrique MIDI 256 échantillons mesurée lors de S790.

| Métrique officielle | S708 GuitarSet CAL32 | Nouveau MIDI final |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

### Frame Pitch SHADOW 1024

| Phase | Référence occupation S708 | S790 tête spécialisée | Gain |
| --- | ---: | ---: | ---: |
| FIT80, composition OOF | 79,9293 % | **86,7532 %** | +6,82 pts |
| CAL32 figé | 77,8353 % | **85,1177 %** | +7,28 pts |

Politique retenue uniquement sur FIT : `cut0.45_harm0.00` (une seule tête réellement entraînée, neuf politiques de seuil et pénalité).

CAL32 par registre : E2–B2 **79,28 %**, C3–B3 **85,61 %**, C4–E5 **86,01 %**. Par texture : accompagnement **83,82 %**, solo **88,27 %**. Par composition : BN2 **84,58 %**, Jazz2 **85,67 %**.

Le score SHADOW décrit des masques par pitch sur la grille 1024, **pas** un flux NoteOn/NoteOff complet à 256 échantillons. Il ne prouve aucune amélioration officielle.

## Comparaison à S708

S708 demeure l'unique référence officielle. S790 montre une amélioration expérimentale cohérente de la tête Frame Pitch à 1024, sans nouvelle inférence MIDI finale. Les valeurs officielles restent donc Onset 74,26 %, Exact-K 62,05 %, Active Exact-K 56,28 %, Offset 77,33 %, Full Note 57,13 % et Frame Pitch 76,46 %.

## Audits

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | PASS | [Run #39277971956](https://github.com/Andriamarosoa/note/actions/runs/39277971956), `S790_AUDIT_data` |
| Acoustique | PASS | [Run #39277971956](https://github.com/Andriamarosoa/note/actions/runs/39277971956), `S790_AUDIT_acoustics` |
| Architecture | PASS | [Run #39277971956](https://github.com/Andriamarosoa/note/actions/runs/39277971956), `S790_AUDIT_architecture` |
| Apprentissage | PASS | [Run #39277971956](https://github.com/Andriamarosoa/note/actions/runs/39277971956), `S790_AUDIT_training` |
| Métriques | PASS | [Run #39277971956](https://github.com/Andriamarosoa/note/actions/runs/39277971956), `S790_AUDIT_metrics` |
| LIVE | PASS | [Run #39277971956](https://github.com/Andriamarosoa/note/actions/runs/39277971956), `S790_AUDIT_live` |

Consensus S790 : SUCCESS, les six audits ont rendu leurs verdicts distincts, aucune autorisation de promotion automatique.

## Verdict

**Terminé, six audits PASS, candidat SHADOW 85,12 % mais non promu.** S790 doit être conservé comme meilleure base expérimentale Frame Pitch connue par rapport à S791 (82,21 % en CAL32 SHADOW), en attendant comparaison sur échantillons identiques et intégration MIDI LIVE 256.

## Prochaine action

Reproduire le candidat en gardant les autres cinq caractéristiques figées, proposer une adaptation causale par fragment/hauteur compatible NoteOn/NoteOff, puis évaluer les six métriques sur le même flux MIDI. Le score S790 isolé ne justifie pas une modification automatique du modèle de référence.
