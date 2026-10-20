---
serie: S803
statut: en_cours
type: experience
reference: S708
branche: codex/note-s803-causal-current-frame-gates
run_id: 39361632654
commit_sha: 1cd2c3d8c6e4795f1c4ce4425cf7211c426e26ac
promotion: false
---

# S803 — audit expérimental de trois gates de têtes par frame/pitch

## Objectif

Continuer NOTE vers **plus de 80 % sur les six mesures** sans régression globale ni composition FIT, en laissant S708 comme référence officielle. Évaluer trois garde-fous acoustiques **strictement causaux** par fragment/hauteur, sur **vraies sorties GuitarSet FIT80**, plutôt que des gains théoriques ou des simulations.

## Hypothèse

Les collisions S802 incluent de nombreux vrais NoteOn ; leur veto aveugle détruirait le rappel. Des caractéristiques présentes à l'instant de décision (`on`, `on_rise`, `harmonic_parent` et probabilités des têtes) pourraient limiter les faux positifs sans utiliser les attaques natives futures. Cette séparation était une hypothèse, non une certitude. Les trois seuils ont été fixés *avant* la lecture des étiquettes de vérité S802.

## Modifications

Branch `codex/note-s803-causal-current-frame-gates`, issue de S802 HEAD `67a49ed`, [PR #107 DRAFT](https://github.com/Andriamarosoa/note/pull/107). Trois règles shadow indépendantes `rise_attack` (adaptive plus on >=0,20 et rise >=0,12), `parent_conflict` (consensus avec parent <=0,02 ou rise >=0,20), `agreement_attack` (adaptive avec general >=0,96, transient >=0,93, rise >=0,10). Modèles S795 et cinq têtes spécialisées conservés ; sélection variant à chaque frame/pitch, aucun remplacement des architectures par un décodeur externe. Script `scripts/s803_causal_frame_replay.py`, test de causalité `test/test_s803_causal_frame_replay.py`, six audits `scripts/s803_six_role_audits.py`, workflow `.github/workflows/note-s803-causal-frame-gates.yml`. Le premier run a révélé un import PyTorch inutile dans les auditeurs ; correctif commit `1cd2c3d`, **replay scientifique non modifié**.

## Protocole

Vraies archives signées S800 #39352880035 (80 traces OOF à cinq cartes), S801 #39354733640 et S802 #39359182007 ; vérification SHA256, contrôle des 12 compteurs FIT S795, replay complet 256 MIDI et six scores `s704_joint_metrics` sur tous les événements réellement émis. Stratégie strictement **FIT** pour l'exploration : aucun CAL32, HELD ni TEST ; score CAL32 S708 non comparé comme s'il s'agissait du même split. Comparer chaque variante à source native FIT sur les six scores et les cinq compositions. AUC rétrospectif des étiquettes S802 par caractéristique, avec bootstrap par piste, **sans intégrer une vérité future au décodeur**.

## Audits obligatoires

Six rôles exécutés comme jobs CI distincts et parallèles : Données, Acoustique, Architecture, Apprentissage, Métriques et LIVE. Ils vérifient en particulier parité, séparation FIT, présence cinq têtes dynamiques et zéro lookahead/hold/source-retiming. Un vrai conseiller NVIDIA NIM `z-ai/glm-5.3-flash` doit s'exécuter en parallèle et un job final vérifier le **7/7** avec les artefacts de la même exécution. L'avis NVIDIA est scientifique et consultatif, jamais la preuve d'un gain F1, ni une autorité de promotion.

## Statut

Le [run #39361632654](https://github.com/Andriamarosoa/note/actions/runs/39361632654) a reproduit 80/80 vraies pistes avec **six audits locaux PASS**. Les trois variantes montrent des régressions et sont **rejetées**. Aucune promotion, S708 intacte. Le conseiller NVIDIA a échoué à répondre à sa première tentative dans ce run ; ses seuls jobs en échec ont été relancés, et le verdict 7/7 est **encore en attente**. Voir `resultat.md` pour toutes les valeurs et AUC mesurées. Aucun test de latence physique LIVE n'est revendiqué.
