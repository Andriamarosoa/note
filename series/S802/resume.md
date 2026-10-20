---
serie: S802
statut: en_cours
type: experience
reference: S708
branche: codex/note-s802-synthetic-native-retrigger-forensics
run_id: en_attente
commit_sha: en_attente
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

En cours — résultats du job GitHub S802 attendus. Cette série est uniquement forensique et ne peut ni modifier la référence S708 ni annoncer 80 %.
