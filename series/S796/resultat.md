---
serie: S796
statut: en_cours
type: experience
reference: S708
branche: codex/note-s796-retrigger-forensics-no-extra-latency
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S796 — État et preuves

**S795 run scientifique** [39340511467](https://github.com/Andriamarosoa/note/actions/runs/39340511467), artefact `S795_ACTIVEK_MULTIHEAD_REAL_MIDI256_FIT80_CAL32` : FIT80 et CAL32 parcourus ; contrôle CAL signé échoué de +1 trame correcte K, six audits S795 non exécutés.

| Métrique FIT80 | Source | Meilleur Active-K candidat S795 |
|---|---:|---:|
| Onset F1 | 73,2520 % | 68,7671 % |
| Exact-K | 61,6972 % | 62,7075 % |
| Active Exact-K | 57,0195 % | 58,2925 % |
| Offset accuracy | 69,8862 % | 61,5546 % |
| Full Note F1 | 50,6528 % | 41,8761 % |
| Frame Pitch F1 | 78,7278 % | 79,6088 % |

Candidat `mask31_cut0.85` rejeté : +1,27 point Active-K, mais −8,78 points Full Note et −8,33 points Offset. Tous les 24 candidats non-identité échouent à la non-régression FIT. S795 doit garder l'identité S708.

**S796 nouveau garde anti-réarmement simultané** : test unitaire ajouté ; résultat CI en attente. Aucun gain S796 déclaré, six scores officiels **non mesurés** ; mesure matérielle de latence **UNKNOWN**. Statuts des six audits : données UNKNOWN, acoustique UNKNOWN, architecture UNKNOWN, entraînement UNKNOWN, métriques UNKNOWN, LIVE UNKNOWN. Pas de promotion.
