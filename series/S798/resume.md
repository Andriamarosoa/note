---
serie: S798
statut: termine
type: infrastructure
reference: S708
branche: codex/note-s798-kimi-independent-adviser
run_id: 39346121986
commit_sha: abbe36b335be4770a994292a04b59bd02f956f43
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

Terminé : tests unitaires simulés et les six audits GitHub Actions PASS, consensus PASS sur le [run S798 #39346121986](https://github.com/Andriamarosoa/note/actions/runs/39346121986), SHA scientifique `abbe36b335be4770a994292a04b59bd02f956f43`. Le job Kimi était SKIPPED dans le run initial. Un essai explicite ultérieur ([run #39346769099](https://github.com/Andriamarosoa/note/actions/runs/39346769099)) a confirmé six audits PASS et consensus PASS, mais le septième job a échoué sur `NVIDIA_API_KEY secret missing`. Aucun appel NVIDIA ni avis externe ; aucun modèle promu. Le déclencheur one-shot a été retiré après le test. Nouvelle tentative diagnostique : [run #39347743466](https://github.com/Andriamarosoa/note/actions/runs/39347743466), six audits PASS et consensus PASS ; Kimi FAIL : `NVIDIA_API_KEY is missing or malformed`. Une valeur de secret est présente mais ne satisfait pas le format `nvapi-` du client. Aucun appel réseau effectué et aucune réponse Kimi reçue. Déclenchement temporaire supprimé. La clé NVIDIA d'origine doit être révoquée. Mise à jour de visibilité : le dépôt NOTE est désormais **privé**, vérifié directement par l'API GitHub après le run. Ce changement a été effectué hors des actions disponibles via cette connexion.

## Vérification finale Kimi K3 — 20 octobre 2026

Le run [#39348737267](https://github.com/Andriamarosoa/note/actions/runs/39348737267) valide six audits PASS + consensus PASS. La nouvelle clé est reconnue par le client, et le fournisseur NVIDIA répond **HTTP 410 Gone** sur le modèle Kimi K3 ; aucun avis de modèle reçu. La documentation NVIDIA Build indique l'endpoint gratuit Kimi K3 comme déprécié et distingue les endpoints partenaires. La sélection d'un autre fournisseur/modèle requiert une décision explicite, pas de substitution automatique. Aucun changement de référence S708 ni des métriques. Déclenchement one-shot retiré.
