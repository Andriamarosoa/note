# S75 — Localisation vérifiée du fold 3 / player05, correction de provenance

**Date : 2026-10-11.** Ce document répond à la recherche des données d'évaluation et corrige la prétention antérieure selon laquelle le fold 3 serait introuvable et vierge d'évaluation. Il ne constitue pas un nouveau score.

## Source exacte des enregistrements et annotations

**GuitarSet v1.1.0** — [Zenodo record 3371780](https://zenodo.org/records/3371780)
- Audio mono pickup mix, **683,1 MB** : [audio_mono-pickup_mix.zip](https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip?download=1), MD5 `aecce79f425a44e2055e46f680e10f6a`.
- Annotations JAMS, **39,1 MB** : [annotation.zip](https://zenodo.org/records/3371780/files/annotation.zip?download=1), MD5 `b39b78e63d3446f2e54ddb7a54df9b10`.
- Le code du dépôt (`src/causal_note/guitarset.py`) charge directement ces ZIP sous `data/GuitarSet/`. Par défaut `ALLOWED_PLAYERS=00..04`; le validateur `scripts/evaluate_v273_fresh_player05.py` utilise spécifiquement le joueur 05 (**60 pistes**), sans l'inclure dans l'apprentissage historique.
- [Workflow reproductible des données](https://github.com/Andriamarosoa/note/blob/codex/v273-scope-rescue/.github/workflows/v273-fresh-player05.yml) : téléchargement, MD5, restauration des poids et score.

## Données natives fold 3 déjà archivées

- [Release native](https://github.com/Andriamarosoa/note/releases/tag/v273-native-decay-35099819151), asset [v273-native-paired-fold-3.zip](https://github.com/Andriamarosoa/note/releases/download/v273-native-decay-35099819151/v273-native-paired-fold-3.zip), **41 139 502 bytes**, contenant la reconstruction native fold3.
- [Audit historique fold3](https://github.com/Andriamarosoa/note/releases/tag/v273-control-fold3-audit-36106051196), 15 279 décisions de reconstruction, 1 969 événements poly et 836 corrects pour **un modèle V27.3 reconstruit distinct** ; cela n'est pas le score de S72.

## Résultat indépendant player05 déjà publié (NE PAS CONFONDRE AVEC S72)

- [Release v273-fresh-player05-37552019921](https://github.com/Andriamarosoa/note/releases/tag/v273-fresh-player05-37552019921), publiée 2026-10-07, `predictions-before-labels.npz`, `scored-predictions.npz`, `report.json`, `report.md`.
- Le test est valide pour **son propre modèle préfigé** : `robust base` 84,268 % global / **39,057 % poly** ; `enriched p≥0.59` 84,152 % global / **38,113 % poly**, 6 corrections /21 régressions (gain −15), validation **échouée**. Une analyse de dérive des descripteurs du joueur 05 a ensuite été exécutée (`scripts/audit_v273_player05_feature_domain_shift.py`).
- **En conséquence, fold3/player05 est à présent une cohorte historiquement consultée** : elle peut servir au diagnostic S72 mais plus de validation finale entièrement aveugle.

## Ce qui manque réellement pour tester S72 sur player05

S72 est un empilement de prédictions S69→S70→S71→S72. Les artefacts S70–S72 publiés contiennent des **prédictions hors-fold sur 0/1/2/4**, non un modèle S72 préentraîné pour fold3. Les 12 réseaux PR16 donnent des sorties OOF 0/1/2/4, pas les probabilités neurales requises pour ces 60 pistes player05. On ne doit ni réutiliser les vieilles prédictions d'autres architectures, ni fabriquer 3 202/7 385 comme score player05.

**Étape correcte** : geler toutes les hyperparamètres actuels ; produire un ensemble de producteurs PR16 entraînés exclusivement sur 0/1/2/4, puis les nouvelles distributions K par fragment sur les audios player05, entraîner S70/S71/S72 avec exclusion totale du joueur 05, rejouer le même pipeline et comparer au parent baseline sur **les mêmes vrais fragments**. Ce serait un diagnostic de transfert, **non une validation indépendante finale** après exposition historique de player05. Pour promotion nouvelle référence, réserver un corpus/jeu de compositions vraiment nouveau, fixer toutes les sources avant de découvrir ses labels.

**Référence de production inchangée** : `freeze_local_combo`. Candidat de développement S72 et ses résultats restent les mêmes, sans validation de nouvelle musique.
