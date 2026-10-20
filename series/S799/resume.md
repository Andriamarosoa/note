---
serie: S799
statut: termine
type: infrastructure
reference: S708
branche: codex/note-s799-nvidia-kimi-k25-adviser
run_id: 39351032882
commit_sha: 428950c46e995754b42af3522ed90a0738bc6a5d
promotion: false
---

# S799 — septième avis scientifique NVIDIA réellement exécuté

## Objectif

Obtenir un septième avis indépendant sur les séries NOTE S795, S796 et S797, sans changer S708, la transcription MIDI causale ni sa latence. Vérifier réellement l'accès au fournisseur et préserver les six audits systématiques. Objectif scientifique à long terme : au moins 80 % sur Onset F1, Exact-K, Active Exact-K, Offset accuracy, Full Note F1 et Frame Pitch F1, sans régression.

## Hypothèse

Un modèle LLM externe, disponible gratuitement depuis NVIDIA NIM, peut identifier des pistes d'ablation, des risques méthodologiques et des difficultés acoustiques, mais ne peut valider ni les notes MIDI ni les métriques. Le modèle utilisé doit être attesté par une réponse HTTP réelle et une sortie archivée, pas seulement figurer au catalogue.

## Modifications

Le client `scripts/s799_nvidia_adviser.py` conserve une allowlist stricte de cinq résumés S795-S797 avec contrôles anti-secrets et sans audio ni code source complet. Après un test initial, deux modèles anciens, Kimi K2.5 et Llama 3.3, se sont avérés retirés (HTTP 410). Le client a été configuré pour les modèles récents `z-ai/glm-5.3-flash` (prioritaire) et `z-ai/glm-5.3` (secours sur HTTP 404/410 uniquement). Le premier a répondu avec succès : aucune substitution secondaire n'a eu lieu. Tous les jetons restent dans GitHub Secrets.

## Protocole

Les six audits sont exécutés comme jobs indépendants puis leurs résultats doivent passer le consensus fail-closed. La septième consultation n'est déclenchée que par commande explicite et ne dispose d'aucune autorité de promotion. Un déclencheur exceptionnel de commit a été temporairement employé pour ce test, puis supprimé ; `workflow_dispatch` est la seule condition conservée dans le workflow final. Les tests unitaires fournisseurs sont mockés, sans appel réseau.

## Audits obligatoires

Les six rôles *Données, Acoustique, Architecture, Apprentissage, Métriques et LIVE* sont tous PASS lors du [run #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882) ; leur consensus est également PASS. La septième sortie externe est archivée comme conseil, pas comme métrique de transcription. Les garde-fous empêchent la promotion de modèles sans scores réels, parité S708 et contrôle de latence causale.

## Statut

**Terminé, infrastructure validée**. Le [run NVIDIA #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882), commit `428950c46e995754b42af3522ed90a0738bc6a5d`, a reçu la réponse du modèle `z-ai/glm-5.3-flash` : 4 617 tokens d'entrée, 2 024 de sortie (6 641 au total). Aucun code de transcription LIVE modifié et aucun des six scores officiels S708 amélioré ou remesuré. Les recommandations du conseiller, résumées dans `series/S799/avis_externe.md`, restent non vérifiées expérimentalement.
