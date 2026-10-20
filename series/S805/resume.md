---
serie: S805
statut: en_cours
type: experience
reference: S708
branche: codex/note-s805-fit-crossfold-onset-risk
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S805 — têtes de risque OOF inter-compositions pour faux NoteOn

## Objectif

Réduire les fausses attaques synthétiques en contexte harmonique, surtout sur GuitarSet SS1/Jazz1, sans diminuer les bonnes réattaques ni dégrader les six métriques ou introduire le moindre maintien temporel/latence. Le précédent S802 a observé sur le consensus 136 attaques additionnelles (74 TP, 62 FP) dont sur SS1 13 TP et 33 FP. Le masque analytique `sustain_rise>=.045 OR harmonic_parent>=.30`, simulé seulement sur les 136 observations S802, éliminerait 61 attaques **dont 32 vraies** : son rejet direct serait dangereux.

## Hypothèse

Deux têtes de risque OOF entraînées sur les quatre autres compositions, sans accès aux annotations de la cible, pourraient distinguer les faux NoteOn de SS1/Jazz1 sans supprimer les vrais. Hypothèse à tester, non revendiquée.

## Modifications

Les scripts S805 ajoutent deux spécialistes de risque indépendants et un contrôle acoustique fixe, sans éditer le moteur LIVE ni les événements S708. Les trois modèles NVIDIA sont configurés pour un audit profond manuel, jamais automatique à chaque push.

## Architecture

S805 préserve les cinq têtes causales indépendantes déjà entraînées (general, bass, polyphonic, transient, harmonic) et leur sélection dynamique pour chaque fragment/hauteur, ainsi que le décodeur MIDI256 sans modification. Deux **têtes de risque supplémentaires** sont entraînées, chacune indépendamment pour chaque composition laissée de côté, sur les seules observations S802 des quatre autres compositions. L'expert `risk_acoustic` utilise uniquement le pitch, l'onset, le sustain, l'offset, les variations onset/sustain et les parents harmoniques actuels ; `risk_joint` ajoute seulement les cinq prédictions OOF indépendantes **courantes**. Leurs seuils de 0,30 sont gelés et ne sont pas sélectionnés sur le FIT d'évaluation.

Trois bras S805 sont pré-déclarés : `spectral_balance` (guard actuel on-off/sustain-rise/parent), `risk_acoustic` (consensus + nouveau head), et `risk_joint` (adaptive + head acoustique et avis des cinq experts). Le classifieur simple, standardisé `LogisticRegression(C=.25,class_weight="balanced")`, est entraîné sur les observations S802 positives/négatives de **quatre compositions exclusivement**, séparément pour chaque cinquième. Aucune annotation de la composition évaluée n'est utilisée pour entraîner son head. Les futures raisons de clôture et le matching vérité restent hors features de décodage.

## Protocole

Rejouer les 80 pistes authentiques FIT OOF enregistrées S800, leurs 5 têtes, les observations S802 et le témoin six-métriques S803 avec vérification SHA256 par piste. Tous les MIDI sont réellement produits par le décodeur S795 original (hop MIDI256). Pour chaque politique : compter les 12 compteurs natifs, calculer Onset F1, Exact-K, Active Exact-K, Offset Accuracy, Full Note F1 et Frame Pitch F1, puis vérifier les écarts du global **et de chacune des cinq compositions**. Toute diminution même ponctuelle sur l'une des 30 comparaisons locales entraîne un rejet du candidat. Aucun gain ne peut être proclamé en CAL32/HELD/TEST, qui restent fermés ; la référence S708 demeure officielle.

## Audits obligatoires

Six audits déterministes (Données, Acoustique, Architecture, Apprentissage, Métriques, LIVE) accompagnent chaque run scientifique complet. Les trois conseillers NVIDIA et leur consensus neuf voies ne sont consultés qu'en exécution manuelle, afin d'éviter les échecs et doublons S804.

## Gestion des runs

Pour éviter les coûts et échecs répétés de S804 : workflow `.github/workflows/note-s805-crossfold-risk.yml` avec smoke seul par push, sauf un commit explicitement marqué `[run-fit80]` ou un dispatch volontaire. Les six audits déterministes (data, acoustics, architecture, training, metrics, live) s'exécutent pour chaque replay réel. Les **trois modèles NVIDIA** sont offerts en parallèle des six audits **seulement** si le workflow est lancé manuellement via `workflow_dispatch`, avec dossier complet FIT/code et consensus strict 9/9. Aucune absence de revue NVIDIA ne permet d'appeler le résultat 9/9 PASS, même si six audits passent.

## LIVE et non-régression

Aucun code LIVE de production, aucun événement S708, aucun NoteOff, aucune durée ou traitement après attaque n'est déplacé. Les masques ne voient que la trame présente et aucune annotation de vérité de la composition évaluée. Le coût de deux nouvelles inférences logistiques n'est **pas encore mesuré sur iRig/Windows** : même un candidat FIT sans régression restera SHADOW tant que ses coûts CPU et la latence physique stricte ne sont pas prouvés.

## Statut

S805 démarré, résultat scientifique FIT80 attendu du workflow GitHub. Les deux fichiers `resume.md` et `resultat.md` sont créés et suivis. Aucun résultat ni amélioration inventés.
