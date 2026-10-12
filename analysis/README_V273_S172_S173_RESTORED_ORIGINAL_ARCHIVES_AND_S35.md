# S172–S173 — Restauration des archives et 11e famille S45 expérimentale

**Branche** : `codex/v273-scope-rescue`. **Référence officielle inchangée** : `freeze_local_combo`. **S80 non promu.**

## S172 — archives originales reconstruites, pas encore les producteurs WAV

[Exécution réussie S172](https://github.com/Andriamarosoa/note/actions/runs/38205875589).

Le script `scripts/s172_recover_original_missing_s45_archives.py` a reconstitué et contrôlé, à partir des fichiers historiques versionnés et des SHA-256 :
- le JSON de modèles originaux S1 acoustic (`series1/models.json.gz`) ;
- les modèles entraînés historiques compressés S3 repair et S4 repair (fragments `models-xz.joblib.part*`) ;
- les preuves de routage S2, S3, S4 (`router-proofs.json`).

Important : `S4_repair` des boucles de régression **n'est pas** `S4_control111` requis par `S5_original527`. Restaurer les modèles historiques n'implique pas qu'ils produisent les 255 groupes requis sur de nouveaux WAV.

## S173 — exécution de vrais checkpoints S35 sur les 57 événements GOAT

[Exécution réussie S173](https://github.com/Andriamarosoa/note/actions/runs/38206160849).

Le script `scripts/s173_original_s35_weights_new_goat.py` exécute, sans entraînement supplémentaire :
- **36 véritables modèles acoustiques S35** : les 9 experts sauvegardés sur les 4 folds originaux ;
- **72 véritables modèles de risque Q S35**, architecture 115D, réévalués après chaque décision dans le routeur à 3 passages ;
- **209 modèles Ridge S45 originaux** (19 pour chacune des 11 familles) avec consensus recalculé à onze sources.

Pour un morceau inédit, l'agrégation des 4 modèles d'experts et Q est **une nouvelle règle** ; le vrai S29 A/B n'étant pas disponible sur les nouveaux WAV, le système utilise le S20 récurrent original comme **adaptateur inédit**. Les poids S35 sont authentiques, mais la proposition n'est donc **pas** une reproduction bit-identique de la source S44.

Sur **57 événements GOAT** :
- S35 propose **0 changement K** par rapport au parent S18 adapté ;
- S45 avec 11 familles garde **1 action sélectionnée** par rapport au S18 adapté ;
- ni le vrai K ni les MIDI ne sont lus ; aucune correction n'est donc prouvée.

## Bloquants à la promotion

Il reste cinq familles sans exécution adaptée sur les nouveaux WAV : `S1_acoustic`, `S2_source`, `S3_repair`, `S4_repair`, `historical_repair_coherent`. Les modèles d'origine S1/S3/S4 sont retrouvés, mais il manque la reconstruction de leurs propositions de groupes et de leurs parents sur audio neuf.

Il manque aussi les véritables caractéristiques S5 527D, dont le constructeur 416D, la chaîne historique complète, le routage de déploiement S69–S80, puis un nouveau corpus polyphonique totalement indépendant pour comparer global/Exact-K2+ à `freeze_local_combo`.

**Ne pas modifier l'officiel avant une preuve d'amélioration indépendante de l'Exact-K global et polyphonique sans régressions non résolues.**
