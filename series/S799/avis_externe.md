# S799 — synthèse vérifiée de l'avis externe GLM‑5.3‑Flash

Source : modèle tiers `z-ai/glm-5.3-flash`, NVIDIA NIM, [run #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882), réponse réelle de 2 024 tokens. L'artefact JSON complet est conservé dans les artefacts GitHub Actions de ce run.

> Cette analyse est une proposition de recherche d'un modèle externe, pas un fait expérimental supplémentaire. Aucune métrique GuitarSet n'a été recalculée par ce conseiller.

## Diagnostic fourni par le conseiller

S795 : le meilleur masque gagnait environ **+1,27 point** Active Exact-K FIT80, au prix d'environ **−8,78 points** Full Note F1, **−8,33 points** Offset, et **−4,48 points** Onset. Le défaut de parité signée `78367` au lieu de `78366` bloque la validation CAL. S796 : correction logique anti-réarmement `closed_this_frame`, sans preuve de gain musical. S797 : le routeur par fragment et hauteur a passé des tests synthétiques, mais les archives S795 n'ont pas les probabilités par fragment nécessaires pour l'évaluer honnêtement sur GuitarSet ; certaines politiques produisent le même score synthétique et ne sont pas discriminées.

## Hypothèses et ablations classées par le conseiller

1. **H1 — Capturer les probabilités OOF par fragment/hauteur**, événements MIDI256 et identifiants de politiques pendant FIT80, sans réentraîner ou modifier les modèles simplement pour archiver. Condition préalable à toutes les ablations et au choix causal par fragment.
2. **H4 — Séparer les décisions de veto Onset et Offset** : empêcher que les gains Active Exact-K proviennent d'une coupure abusive des NoteOff. Comparer offset guard seul, onset guard seul, sans guard et combinaison, avec tests anti-réarmement S796.
3. **H3 — Conditionner le veto Onset à une attaque native présente sur la frame courante** pour ne pas effacer les attaques faibles. Ne jamais utiliser de frame future.
4. **H2 — Examiner les confusions harmoniques / octaves**, notamment les couples f/2f et f/3f, mais protéger les octaves réellement jouées ; conditionner les exclusions à l'absence d'attaque autonome.
5. **H5 — Sélectionner dynamiquement les familles de têtes par fragment/hauteur** sur plis OOF, sans nombre fixe arbitraire. Aucune sélection sur CAL32, HELD ou TEST.

## Contrôles exigés

Exiger la parité S708 exacte, d'abord sur FIT80, puis confirmation CAL32 des six métriques sans régression. Pas de tuning sur CAL. Comparer coût CPU apparié médiane et p95, contrôler la latence physique et tester l'indépendance des décisions passées vis-à-vis de toute entrée audio future. Aucun hold, lookahead, buffering, retiming ou différé NoteOff n'est permis.

## Décision expérimentale

S799 valide **le branchement du septième conseiller**, pas le progrès de la détection musicale. Piste prioritaire pour la série S800 : récupération des probabilités et décisions OOF réelles avant de lancer des recherches additionnelles de masques. Aucune promotion S708.
