---
serie: S801
statut: termine
type: experience
reference: S708
branche: codex/note-s801-fit-fragment-metric-ablations
run_id: 39354733640
commit_sha: ce470b37a0cc5c21533086cd46e3793a9490b7a3
promotion: false
---

# S801 — ablations réelles des sélecteurs par fragment et hauteur

## Objectif

Exécuter les ablations causales H2/H3/H4/H5 recommandées par le 7e conseiller et par les audits NOTE, sans changer les architectures déployées. Objectif à long terme : six mesures Onset F1, Exact-K, Active Exact-K, Offset accuracy, Full Note F1 et Frame Pitch F1 à plus de 80 %, sans régression. Cette série mesure d'abord les vraies sorties FIT80 OOF, sans ouvrir CAL32.

## Hypothèse

Les cartes OOF S800 permettent de comparer les cinq sélecteurs S797 et trois gardes causaux spécialisés, en mesurant les notes MIDI256 réellement émises. Les gains synthétiques de S797 ne prouvaient rien sur GuitarSet. Le meilleur sélecteur doit préserver toutes les six métriques FIT, composition par composition, sans attendre une future attaque ou prolonger un NoteOff.

## Modifications

`scripts/s801_fit_fragment_ablations.py` évalue les huit politiques immuablement déclarées : identité, consensus, veto harmonique, poly-gap, adaptive, garde attaque native, garde création Offset et protection parent harmonique. Le décodeur MIDI de S795 reste inchangé ; seuls les choix immédiats des probabilités par tête et fragment sont filtrés. Les nouvelles têtes n'ont pas de nombre fixe imposé et n'ont pas été réentraînées en S801.

## Protocole

Les 80 traces OOF S800 et leurs signatures SHA256 doivent être vérifiées avant lecture ; les six métriques sont recalculées avec `scripts.s704_joint_metrics` sur tous les MIDI256 réels. Chaque composition est évaluée séparément. Le contrôle identité reproduit exactement les comptes FIT S795 avant tout examen des autres politiques. Le seuil FIT anti-régression est strict pour chacune des six mesures et chaque composition. CAL32 reste scellé ; une victoire sur FIT n'est pas une validation CAL ni une promotion. Aucun ajout de buffering, retard de NoteOff, hold ou frame future. L'algorithme supplémentaire n'a pas encore été profilé sur l'appareil réel, donc non déployable LIVE.

## Audits obligatoires

Données : cinq plis tenus à part et aucun CAL. Acoustique : attaques, extinctions, octaves et harmoniques. Architecture : décisions par fragment. Apprentissage : aucune réexécution des banques. Métriques : six nombres réels appariés et veto de toute régression. LIVE : absence de lookahead et contrainte de zéro délai ajouté ; latence CPU physique encore non mesurée. Six audits sur les preuves effectives, puis consensus strict.

## Statut

**Terminée — évaluation FIT80 réelle et six audits PASS.** Run [#39354733640](https://github.com/Andriamarosoa/note/actions/runs/39354733640), commit `ce470b37a0cc5c21533086cd46e3793a9490b7a3`. Les huit politiques ont toutes été évaluées sans nouvel entraînement depuis les 80 traces OOF signées de S800. Tous les audits et leur consensus sont PASS, mais **aucun candidat non identitaire ne franchit les six garde-fous** : sept alternatives rejetées, référence S708 non modifiée. L'analyse des événements met en évidence 95 fermetures lors de réattaque native pour 136 attaques additionnelles (consensus), 81/118 (adaptive), 61/90 (poly-gap). Aucune latence physique mesurée ou promotion.
