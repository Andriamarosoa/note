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

# S790 — Résumé

## Objectif

Tester une première tête Frame Pitch réellement indépendante sur 49 hauteurs, avant de comparer ultérieurement son apport avec une banque d'experts hétérogènes. Préserver intégralement S708/S707, les événements MIDI finaux et la causalité LIVE.

## Hypothèse

Une tête Sustain/Pitch apprenant des caractéristiques causales spectrales et harmoniques peut augmenter l'occupation correcte des notes, particulièrement des fondamentales graves et des accompagnements, sans accéder aux annotations pendant l'inférence.

## Modifications

- `scripts/s790_independent_framepitch_expert.py` : tête logistique 40D indépendante, 49 hauteurs, neuf politiques FIT-only à seuil et pénalité harmonique.
- `test/test_s790_independent_framepitch_expert.py` : tests d'occupation à partir des événements émis, causalité préfixe et géométrie F1.
- `scripts/s790_six_role_audits.py` : six audits indépendants, source S708 contrôlée.
- `.github/workflows/note-s790-framepitch.yml` : FIT80/CAL32, métriques de tête SHADOW, consensus bloquant.
- Les deux fichiers obligatoires `resume.md`, `resultat.md` retracent les résultats et limites.

## Protocole

GuitarSet FIT80 sur cinq compositions pour le choix OOF, puis CAL32 uniquement pour le diagnostic gelé ; HELD/TEST non ouverts. Six jobs d'audit distincts et consensus avant toute validation documentaire. **Sortie évaluée = masque de notes actives à décision 1024, pas le MIDI final hop 256.** S708 reste la référence officielle.

Le modèle FIT sélectionné est `cut0.45_harm0.00`. Valeurs mesurées SHADOW : FIT base 79,9293 %, FIT sélectionné 86,7532 % ; CAL32 base 77,8353 %, candidat 85,1177 %. Les chiffres ne démontrent pas une amélioration du F1 officiel.

## Audits obligatoires

- Données : signature S708, split FIT/CAL disjoint, HELD/TEST fermés.
- Acoustique : causalité des indices temporels et relations harmoniques.
- Architecture : une tête indépendante et neuf politiques sélectionnées sur FIT.
- Apprentissage : reconstitution des neuf scores FIT et du gagnant.
- Métriques : recompte F1/precision/recall et séparation SHADOW vs MIDI officiel.
- LIVE : aucune modification des événements, du décodeur ou de S707/S708.

## Statut

**Terminé**, [GitHub Actions #39277971956](https://github.com/Andriamarosoa/note/actions/runs/39277971956) SUCCESS, SHA `a56362661f11df109e4cfafdecb07bb38e27a7cc`, six audits et consensus réussis. Nouveau Frame Pitch SHADOW CAL32 **85,12 %**. Aucune promotion scientifique du MIDI final.
