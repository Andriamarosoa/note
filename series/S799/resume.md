---
serie: S799
statut: en_cours
type: infrastructure
reference: S708
branche: codex/note-s799-nvidia-kimi-k25-adviser
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S799 — relecteur scientifique NVIDIA disponible

## Objectif

Remplacer le modèle indisponible `moonshotai/kimi-k3` (HTTP 410 confirmé en S798) par une alternative officiellement référencée et obtenir le véritable septième avis indépendant pour NOTE. Objectif scientifique inchangé : six mesures Onset F1, Exact-K, Active Exact-K, Offset, Full Note F1, Frame Pitch F1 à >=80 % sans perte et sans latence ajoutée.

## Hypothèse

`moonshotai/kimi-k2.5`, disponible au catalogue NVIDIA NIM, peut produire des hypothèses ablatables sur les têtes indépendantes et le routage par fragment ; en cas de 404/410, un repli strictement limité à `meta/llama-3.3-70b-instruct` est autorisé. Le catalogue ne prouve pas la disponibilité réelle ni les quotas, que seul l'appel effectif peut établir.

## Modifications

Le nouveau client `scripts/s799_nvidia_adviser.py` conserve l'allowlist des résumés S795-S797 et le filtrage des secrets S798. Kimi K2.5 utilise le mode instantané NVIDIA `thinking: disabled`, sans `reasoning_effort` K3. Deux modèles maximum. Aucun repli sur erreurs 401, 429, 422. Secret `NVIDIA_API_KEY` injecté exclusivement depuis GitHub Actions, sans journalisation de sa valeur. Réponse du fournisseur conservée séparément comme avis non validé.

## Protocole

Six audits déterministes préalables, consensus fail-closed, puis une seule invocation du conseiller externe S799. Appel externe autorisé par lancement manuel, avec un déclencheur exceptionnel par commit marqué `[run-nvidia-once]` pour ce test : ce déclencheur sera retiré dès sa fin. Les avis ne touchent ni à l'entraînement, ni aux poids, ni à la sortie MIDI, ni au LIVE. Aucun score GuitarSet, aucune latence matérielle supposée.

## Audits obligatoires

Données : seules cinq synthèses autorisées. Acoustique : harmoniques, onsets, offsets. Architecture : deux modèles au maximum, aucun redirect. Apprentissage : test du fournisseur par mocks sans réentraînement. Métriques : toutes les six cibles gelées, aucune promotion. LIVE : branche indépendante de la chaîne microphone, zéro buffering, zéro head latency induite.

## Statut

Expérience S799 en cours. La réussite de Kimi K2.5 ou du modèle de secours reste à confirmer par une réponse non vide du fournisseur. Les six audits NOTE restent obligatoires. La référence officielle S708 CAL32 ne bouge pas.

**Révision après essai #39350745687 :** Kimi K2.5 et Meta Llama 3.3 ont tous deux renvoyé HTTP 410 (fin des endpoints gratuits). Le test continue avec les deux modèles récents affichés « Free Endpoint Available » par NVIDIA Build : `z-ai/glm-5.3-flash` puis `z-ai/glm-5.3` (repli seulement en cas de 404/410). Les paramètres de génération sont adaptés au fournisseur ; les six audits préalables ne changent pas.
