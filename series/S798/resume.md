---
serie: S798
statut: en_cours
type: infrastructure
reference: S708
branche: codex/note-s798-kimi-independent-adviser
run_id: pending
commit_sha: pending
promotion: false
---

# S798 — septième relecteur Kimi via l'API NVIDIA

## Objectif

Rendre utilisable le modèle `moonshotai/kimi-k3` de l'exemple `test.py` comme conseiller scientifique indépendant et non-promouvant pendant les audits NOTE. Réunir des avis sur les blocs Onset, Exact-K, Active Exact-K, Offset, Full Note et Frame Pitch, vers 80 %, sans augmenter la latence.

## Hypothèse

Une critique de modèle externe peut proposer des pistes de segmentation des têtes, de sélection par fragment et de séparation des harmoniques qu'il faudra évaluer sur les mêmes six métriques MIDI256, avec protocole de causalité inchangé. Son avis ne constitue jamais une vérité terrain ni un score.

## Modifications

Un client NVIDIA Kimi lit une allowlist de cinq résumés S795-S797, jamais les secrets, les données audio, les labels HELD/TEST ni le code propriétaire complet. Secret via `NVIDIA_API_KEY`, aucun jeton dans le dépôt. Tests réseau simulés, contrôles stricts de censure et refus de redirections. Un septième job Kimi n'est exécuté que sur demande manuelle `workflow_dispatch` et seulement si la clé *renouvelée* est ajoutée aux secrets GitHub Actions.

## Protocole

À chaque audit, les six rôles NOTE déterministes restent obligatoires. Le job externe est séparé et ne peut pas promouvoir un modèle : ses hypothèses sont archivées pour revue humaine. Première exécution CI par push : tests simulés uniquement, aucun appel réseau externe réel. L'exécution de Kimi nécessite explicitement le nouveau secret et le déclenchement manuel. Aucune modification LIVE. S708 reste référence, aucune données CAL utilisée pour sélectionner les politiques.

## Audits obligatoires

Données : confidentialité de l'allowlist et exclusion des secrets. Acoustique : questions sur harmoniques et erreurs. Architecture : externalisation sans pouvoir de décision et interdiction de redirection. Apprentissage : pas de nouvel entraînement. Métriques : pas de F1 fictif ni autorité externe. LIVE : pas d'appel dans le pipeline causal. Six avis indépendants, puis consensus strict.

## Statut

En cours : infrastructure S798 à valider par CI. La consultation externe n'est pas encore exécutée. La clé incluse dans l'exemple de l'utilisateur est compromise et ne doit pas être réutilisée. Dépôt NOTE toujours public lors de la conception ; le propriétaire doit changer la visibilité dans les paramètres GitHub.
