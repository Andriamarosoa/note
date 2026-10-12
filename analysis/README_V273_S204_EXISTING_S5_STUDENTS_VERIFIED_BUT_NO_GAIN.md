# S204 — Vérification indépendante du remplacement S5 existant, pas de réentraînement

Branche `codex/v273-scope-rescue`. Référence officielle `freeze_local_combo`. S80 non promu.

## Décision

**Ne pas reconstruire une seconde fois l'étudiant S5 : S129–S135 existent et leurs checkpoints sont préservés.** Dans S204, l'étudiant S133 111D et les sept régressions de probabilités S134 sont rechargés depuis l'archive permanente GitHub S136, validée par SHA256. Le code vérifie le contrat source authentique S116 de 57 événements audio : 58 résumés, 46 géométries, 7 one-hot K figés. Il recalcule `S133.predict_proba`, `S134.forward` puis la fusion déterministe S135 et compare aux probabilités archivées de 57 événements, sans labels ni réentraînement.

## Run confirmé

- [S204](https://github.com/Andriamarosoa/note/actions/runs/38215937218) : terminé, succès
- SHA256 checkpoint permanent S133 : `9bb74dac6a394059b619ca3365d381a9fabd4f1b73fc27dc26c74ff936f65a29`
- SHA256 checkpoint permanent S134 : `527979334b3a420a823e1d78d3f1e48560ed5765a84489739d5e2d643e222876`
- S116 111D source SHA256 : `0a78175a37e36c6dba2cc9310cdd400ac2c90f48f68cf886ac964e56e36234ec`
- Erreurs max en relecture : S133 `2.9622e-08`, S134/S135 `2.8768e-08`.
- S135 modifie exactement 2 des 57 sorties. Hist K0–K6 S135 : `[46,9,2,0,0,0,0]`.

La **performance réelle** sur ce GOAT est déjà connue par le run S137, postérieur au gel des prédictions :
`freeze_local_combo` 44/57 global, 2/3 poly ; S135 44/57 global, 1/3 poly, 1 correction et 1 régression, **net 0**. [Rapport S137/S138](./README_V273_S137_S138_FIRST_REAL_GOAT_SCORE.md). Cet échantillon est trop petit pour des conclusions générales, mais **rejette la promotion de S135** sur la preuve existante.

S204 n'a ni découvert le constructeur original S5 527D, ni restauré ses 416 colonnes, ni démontré l'exécution de S45 16 familles, S46 193 politiques et S80 complet. La compatibilité d'interface 111D avec S116 n'est pas une validation du vrai S45/S46 historique. **Ne pas utiliser les résultats d'imitation du professeur comme une exactitude K.**

## Prochain choix rationnel

Pour S80 **historique**, exiger fichier source original S4/S5 `owned527` ou matrice authentique S5 527D avec preuves de provenance, puis rejouer les probabilités historiques événement par événement. Les variantes de reconstruction S118/S123/S125/S201/S202 ne le font pas. Les 255 propositions et 9 audits, eux, ont été restaurés sur les 7 493 événements par S198.

Pour une **nouvelle version expérimentale distincte**, il est possible de réutiliser le checkpoint S133/S134 comme branche de remplacement plutôt que le réentraîner, mais la validation S137 rejette l'étudiant S135 tel quel. Toute nouvelle architecture doit être versionnée explicitement et gelée avant une évaluation pairée sur suffisamment de nouveaux WAV polyphoniques annotés. Les modèles originaux S45/S46 manquants doivent être résolus séparément.

**Référence inchangée. S80 non promu.**
