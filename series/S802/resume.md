---
serie: S802
statut: termine
type: experience
reference: S708
branche: codex/note-s802-synthetic-native-retrigger-forensics
run_id: 39357068590
commit_sha: 90e73bfc69ddab31588ed20e53137c06a213b007
promotion: false
---

# S802 — distinguer les attaques synthétiques utiles des fausses réattaques

## Objectif

Expliquer le facteur limitant des sept politiques non identitaires de S801 : les attaques synthétiques entrent-elles en collision avec des attaques natives vraies, et quelle proportion de ces collisions correspond réellement à une attaque correcte sur GuitarSet ? L'objectif général NOTE de plus de 80 % concerne les six métriques Onset F1, Exact-K, Active Exact-K, Offset accuracy, Full Note F1 et Frame Pitch F1 ; cette série ne propose encore aucun gain ni modèle nouveau.

## Hypothèse

Les 95 fermetures `s795_source_retrigger` du consensus S801 ne sont pas nécessairement 95 faux positifs ; certaines attaques synthétiques peuvent être les véritables attaques, avec une attaque native trop tardive. Un appariement optimal un-à-un aux notes annotées doit révéler la proportion des vrais positifs, fausses attaques et profils acoustiques initiaux, exclusivement hors ligne.

## Modifications

Le module `scripts/s802_retrigger_forensics.py` recharge les cinq cartes OOF et la vérité FIT de chaque piste S800, puis reproduit sans modification le décodeur et trois politiques S801 (consensus, poly_gap, adaptive). Il appaire les NoteOn aux vérités avec la fonction `event_score` authentique (matching hongrois à tolérance 80 ms), reconstruit chaque durée de vie synthétique et la raison du NoteOff, et observe si une source native a attaqué la même hauteur à l'instant de fermeture. Il archive les valeurs acoustiques courantes au moment de l'attaque, les probabilités des cinq têtes, les classes vrai/faux positif et les écarts temporels. Aucune information future n'entre dans le modèle ou les décisions.

## Protocole

Comparer les 12 comptes FIT S800, puis les 12 comptes de chacune des trois politiques aux résultats signés S801. Chaque trace est vérifiée avec SHA256 avant lecture. Les 80 pistes et cinq compositions doivent être présentes. Analyser les sous-groupes par composition, les durées avant réattaque native et les mêmes conditions de vérité d'attaque. CAL32, HELD et TEST restent fermés ; aucun paramètre n'est sélectionné ici.

## Audits obligatoires

Six rôles indépendants, avec fail-closed consensus : Données (partitionnement et vérité hors ligne), Acoustique (5 têtes et durées), Architecture (décodeur/horloges gelés), Apprentissage (aucun oracle dans inférence), Métriques (six mesures identiques à S801 et classes TP/FP sans double compte), LIVE (aucun hold, aucune modification de NoteOff ni latence ajoutée déclarée).

## Statut

**Terminée, analyse forensique vérifiée et non promue** au [run #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590) (commit `90e73bfc69ddab31588ed20e53137c06a213b007`). 80/80 traces FIT80 auditées, trois politiques reproduites avec parité de toutes les métriques S801, six audits GitHub indépendants PASS et consensus PASS. Collisions consensus : 95 fermetures par réattaque native dont **64 NoteOn synthétiques correctement appariées** et **31 fausses selon le matching officiel** ; ne pas supposer que toutes les collisions sont des fausses notes. Toutes les métriques CAL32 et la référence S708 sont inchangées. L'ancien run forensique S802 n'avait pas lancé GLM ; ce manque a été corrigé a posteriori : [run du 7e conseiller #39358229144](https://github.com/Andriamarosoa/note/actions/runs/39358229144), **SUCCESS**, via `z-ai/glm-5.3-flash`, après relecture des six audits S802 d'origine. Le workflow réutilisable `.github/workflows/note-reusable-seventh-adviser.yml` doit désormais être appelé par les futures séries après leurs propres six contrôles. Avis purement consultatif, aucune promotion.

## Septième conseiller automatique

Le workflow S802 scientifique déclenche désormais la consultation NVIDIA **en parallèle** des six rôles d'audit, après la reconstruction des preuves FIT80. Le consensus final doit réussir les six rôles ET vérifier l'artefact de la vraie réponse GLM-5.3-Flash. Aucun `workflow_dispatch` manuel n'est requis pour les exécutions normales. Un échec de NVIDIA ne sera plus enregistré comme septième PASS. La consultation n'est jamais un pouvoir de promotion. Ce dispositif est appelé automatiquement par le workflow S802 ; les nouvelles séries S803+ doivent intégrer le job partagé `.github/workflows/note-reusable-seventh-adviser.yml`.

## Test de bout en bout automatique 7/7 confirmé

Le [run #39359182007](https://github.com/Andriamarosoa/note/actions/runs/39359182007) a validé l'automatisation **sans déclenchement manuel** : six audits S802 PASS et septième relecture NVIDIA GLM-5.3-Flash PASS, lancés en parallèle dès que le job scientifique FIT80 était terminé. Le consensus final `NOTE_SEVEN_OF_SEVEN_PASS S802 z-ai/glm-5.3-flash promotion=false` est PASS ; les artefacts NVIDIA et six rôles sont liés au même run. L'ancien workflow séparé a été supprimé. Les séries S803 et suivantes doivent importer le workflow partagé pour reproduire cette parallélisation sur leurs boucles.
