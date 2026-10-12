# S171 — S163–S170 : S45 passe de 5 à 10 producteurs exécutables, sans promotion

**Branche de recherche** : `codex/v273-scope-rescue`.
**Référence officielle** : `freeze_local_combo`. **S80 non promu**.

## Résultats réellement exécutés

| Boucle | Producteurs S45 authentiques disponibles sur GOAT | Ridge originaux vérifiés | Interventions acceptées par les règles de risque partielles |
|---|---:|---:|---:|
| [S163](https://github.com/Andriamarosoa/note/actions/runs/38198248491) | 5 sur 16 | 95 | 0 |
| [S167](https://github.com/Andriamarosoa/note/actions/runs/38198877007) | 8 sur 16 | 152 | 0 |
| [S169](https://github.com/Andriamarosoa/note/actions/runs/38199247838) | **10 sur 16** | **190** | **1** |

S164 a vérifié **tous les 304** checkpoints originaux Ridge S45 :
[run](https://github.com/Andriamarosoa/note/actions/runs/38198384234).
Ils sont récupérés de l'exécution historique 37891039201, et ne
sont pas réentraînés.

Ces dix familles sont: S11_votes_time_audio, S12_audio_logistic, S19_recurrent_ABA, S20_repeated_ABA, S24_Bfirst, S25_Bfirst, S27_early_stop, S38_morphology, S40_temporal_CNN, S41_neural_fusion.

Les **six familles encore absentes sur nouvel audio**, malgré les
variants S44 et SHA historiques identifiés, sont:
S1_acoustic, S2_source, S3_repair, S4_repair, historical_repair_coherent, S35_multiroute.

Les sources de S19, S20, S24, S25 et S27 reposent sur de vrais
réseaux récurrents enregistrés (S166 et S168). Les 10 familles
partagent toutefois des entrées **S5/S18 adaptées** et leurs
moyennes sur les 19 morceaux d'origine sont une règle de déploiement
nouvelle. La disponibilité des vrais poids ne garantit **pas**
la reproduction bit-exacte des propositions historiques.

## [S170 — intervention originale confirmée, justesse NON connue](https://github.com/Andriamarosoa/note/actions/runs/38199557501)

Événement : `example2_di.wav:161909`.

- Parent S18 adapté : **K1** ; proposition S25 B-first : **K0**.
- Soutiens indépendants dans le catalogue partiel : S11, S12, S24
  et S25 (4 familles dont la famille sélectionnée).
- Correction attendue par Ridge : **0.45647651**.
- Régression attendue par Ridge : **0.36834586**.
- Score net fixé (lambda 1 / seuil 0) : **+0.08813065**.
- S170 reproduit **exactement** l'unique action de S169 à partir des
  vecteurs et risques scellés. Aucune vérité MIDI n'est lue et aucun
  seuil n'a été optimisé sur GOAT.

**Cette action n'est pas une correction vérifiée** : nous ne connaissons
pas son Exact-K réel en validation indépendante. Il manque six
familles, donc la caractéristique de consensus n'est pas encore
celle du S45 original à 16 sources.

Toutes les sorties et preuves négatives S163–S170 sont scellées
avec SHA-256 dans la [prerelease de recherche permanente](https://github.com/Andriamarosoa/note/releases/tag/v273-S163-S170-real-original-S45-10of16-GOAT-research).

## Conditions toujours bloquantes

Le S5 historique 527D n'est toujours pas reproduit sur un WAV nouveau
(constructeur original 416D absent). Un pipeline avec S135 est
**S80-deploy-v1**, pas le S80 historique identique.

Il faut (i) exécuter les six producteurs restants sur des
morceaux inédits, (ii) calculer les votes réels à 16 et les
risques S45 correspondants, (iii) connecter sans labels
S46→S69→S72→S79→S80 (le final K0/K2 exact est déjà connu par S147),
(iv) figer les prédictions et valider sur un corpus polyphonique
song-disjoint réellement indépendant contre `freeze_local_combo`.

Le petit ensemble GOAT ayant été annoté lors d'un essai précédent
ne peut pas constituer la preuve finale après ces reconstructions.

**Aucune promotion officielle ni amélioration indépendante n'est
prouvée par ces boucles.**
