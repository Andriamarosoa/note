# NOTE — Répartition des six missions d'agents (S751, 19-10-2026)

**IMPORTANT : dépôt préparé pour délégation, mais AUCUNE session autonome n'a été lancée dans cette conversation.**

Tentative de création d'issue GitHub avec l'assignee Copilot `copilot-swe-agent[bot]` : **refus GitHub HTTP 422** — `assignees copilot cannot be assigned to this issue`.
La variante `copilot-swe-agent` via l'ajout d'assignee n'a attribué personne après vérification. Ne pas interpréter la présence des issues comme des agents actifs.

## Répartition et ordre

| Agent | Issue | Mission | État |
| --- | --- | --- | --- |
| D | https://github.com/Andriamarosoa/note/issues/77 | parité six métriques, K, fuites et labels | prêt, non attribué ; **préalable aux validations des autres** |
| A | https://github.com/Andriamarosoa/note/issues/78 | LIVE, latence et causalité par fragment | prêt, non attribué |
| B | https://github.com/Andriamarosoa/note/issues/79 | fondamentales graves, DSP, harmoniques superposées | prêt, non attribué |
| C | https://github.com/Andriamarosoa/note/issues/80 | architectures ML, masques dynamiques, K2–K6 | prêt, non attribué |
| E | https://github.com/Andriamarosoa/note/issues/81 | tests, CLI, code, CI, provenance | prêt, non attribué |
| F | https://github.com/Andriamarosoa/note/issues/82 | RED TEAM contradictoire des PR A–E | prêt ; **attendre les résultats A–E** |

## Instructions de démarrage depuis un environnement avec vrais sous-agents

- Fournir à chaque agent son issue, `AGENTS.md` et l'URL des runs.
- **Branche de base obligatoire** : `codex/note-s750-six-axis-cal32-rootcause`. Le `main` existant est obsolète ; ne jamais utiliser `main` comme base d'un correctif S750 sans rebase vérifiée.
- Isoler travail A/B/C/D/E par branche `agents/note-<lettre>-...` et PR DRAFT. D commence par figer le protocole ; F ne revoit qu'après.
- Original S708/S707/S80 gelé, LIVE non modifié. Respect des six scores calculés sur le même MIDI causal, aucun accès TEST et pas de promotion sans protocole signé et seuils non-régression.
- Pas de preuve >80% en l'absence de validation. Pas de distillation avant résultat >80% sur les six métriques.

## État scientifique figé pour délégation

S750 run SUCCESS : https://github.com/Andriamarosoa/note/actions/runs/39207493606.
S708 GuitarSet CAL32 Onset 74,263862%, Exact-K 62,045541%, Active Exact-K 56,276249%, Offset 77,329193%, Full Note 57,131931%, Frame Pitch 76,462635%.
Vrais K2 35,6841% correct, K3 32,9040%, K4 32,6716%, K5 9,3908%, K6 6,3969%. 36 890 trames actives sous-comptées.
S748 terminé, 12/12 boucles, **0 variante promue**. S749 est une expérimentation Offset distincte et ne doit pas être confondue avec un agent autonome.

## Blocage actuel

Le connecteur GitHub actif autorise issues, commits, PR et runs Actions, **mais ne propose pas d'API de lancement de sous-agents Codex/Work**. L'assignation au coding agent Copilot a été refusée.
Pour obtenir de vrais agents parallèles, lancer les missions depuis un environnement disposant de cette capacité (par exemple Work/Codex ou GitHub Copilot coding agent **une fois activé et autorisé**), en choisissant explicitement la branche de base. Une issue créée n'est pas une session lancée.
