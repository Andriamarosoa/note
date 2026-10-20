---
serie: S804
statut: en_cours
type: infrastructure
reference: S708
branche: codex/note-s804-nvidia-three-model-deep-audit
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S804 — trois modèles NVIDIA pour audits approfondis du code et de GuitarSet

## Objectif

Exécuter en profondeur des audits séparés du code et des données réelles GuitarSet, avec six contrôles déterministes indépendants et trois modèles NVIDIA, sans prétendre modifier ni améliorer les six métriques S708, et sans toucher au décodage LIVE. La cible finale NOTE reste plus de 80 % sur Onset F1, Exact-K, Active Exact-K, Offset, Full Note F1 et Frame Pitch F1 **sans aucune régression**.

## Hypothèse

Une lecture des seuls `resultat.md` par un LLM ne suffit pas pour découvrir des lacunes d'architecture, des contaminations de jeux d'essai ou des erreurs de métriques. Un inventaire AST de tous les scripts Python associé à des extraits de code précis et à l'audit local de 80 traces FIT authentiques peut produire un examen extérieur plus ciblé et réfutable. Trois opinions indépendantes spécialisées peuvent se contredire, et aucune ne constitue une preuve de gain F1.

## Modifications

Création d'un pipeline S804 autonome : `scripts/s804_deep_audit_bundle.py` parcourt tous les Python suivis par Git et vérifie numériquement les 80 traces FIT S800 originales ; `scripts/s804_nvidia_deep_review.py` attribue trois modèles distincts (Gemma 4 31B : code/architecture ; NVIDIA Nemotron 3.5 Lightning : données ; Poolside Laguna XS 2.1 : contre-audit). Les tâches partent en parallèle, utilisent des extraits source numérotés et SHA256 avec les données statistiques signées FIT80. `scripts/s804_local_six_audits.py` garde six rôles indépendants et `scripts/s804_nine_consensus.py` bloque si l'une des neuf preuves manque. Aucun nouveau routeur, tête, politique MIDI ou état LIVE de production n'est modifié.

## Protocole

Descendance stricte de S803 validée, base documentaire S708 officielle inchangée. Les 80 fichiers GuitarSet FIT signés doivent être lus et vérifiés avec `verify_snapshot` et leurs cartes de cinq têtes. Rejeter fichiers corrompus, horloges non causales, partitions absentes, non-finitudes et dérives des 12 compteurs (frames=306536, k_correct=189124). Regrouper données par composition, piste et hauteur ; joindre résultats réels S803 et collisions réelles S802. Faire une analyse syntaxique de **tous** les Python suivis par Git, puis sélectionner explicitement les extraits numérotés de 11 chemins critiques. Le modèle externe n'a pas lu tous les fichiers en entier ; l'exhaustivité locale et la couverture externe bornée doivent être distinguées. Les modèles ne reçoivent pas de fichiers audio bruts ni de tables vérité complètes. CAL32/HELD/TEST restent fermés. Toute détection de secret potentiel empêche l'envoi.

## Audits obligatoires

Six jobs CI de données, acoustique, architecture, entraînement, métriques et LIVE, en parallèle de trois relectures NVIDIA NIM. Les trois avis sont liés au même bundle d'évidence et SHA, ont trois identités de modèle distinctes et réclament des contre-exemples, tests et localisations fichier+ligne. Consensus fail-closed **9/9** ; une indisponibilité modèle ne se convertit pas en PASS. Des avis contradictoires sont conservés ; la réussite du pipeline ne signifie pas absence de défaut du code ou amélioration des six métriques. NVIDIA est consultatif, ne peut modifier aucune référence ni accorder une promotion.

## Statut

En cours : workflow scientifique GitHub S804 à valider. Aucune nouvelle mesure CAL32 ni gain six-métriques annoncé. La sécurité LIVE reste inchangée, latence matérielle non mesurée.


## Réduction des exécutions redondantes (2026-10-20)

Audit rétrospectif : sept lancements complets de S804, dont cinq FAIL et deux annulés, après des `push` successifs. Les six audits locaux réussissaient sur les traces réelles ; les avis NVIDIA ont connu des timeouts, réponses trop courtes et faux rejets dus à une expression régulière Python erronée. Cette expression recherchait littéralement des antislashs et non des espaces/points valides, empêchant de reconnaître les chemins source. Corrigée avec test de non-régression, puis contrôle à nouveau dans la porte 9/9.

Nouvelle règle : un `push` exécutera **seulement smoke** (syntaxe, tests unitaires réseau désactivé et documentation). Le run lourd de 80 traces + six audits + trois appels NVIDIA + consensus exige désormais `workflow_dispatch` volontaire sur le SHA à auditer. Aucun nouvel appel fournisseur automatique à chaque commit. Les 9/9 restent obligatoires pour déclarer S804 complète ou promouvoir une expérience ; un run smoke vert ne vaut PAS une réussite 9/9.
