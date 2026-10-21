---
serie: S823
statut: en_cours
type: experience
reference: S708
branche: codex/note-s823-legacy-s795-features-causal256
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S823 — Test des 23 véritables variables S795 sur une horloge MIDI256

## Objectif

Mesurer la contribution des **vraies** 23 variables S795 utilisées en S810, projetées sans anticipation à la cadence MIDI256, pour comprendre la différence entre S810 76,5656% SHADOW 1024 et S822 54,3857% SHADOW 256. Un diagnostic ne vaut pas intégration LIVE.

## Hypothèse

Les 23 variables S795 contiennent des prédictions du modèle S708, sa distribution de K, ainsi que l'état des MIDI déjà émis, en plus des données acoustiques. Une partie du record S810 peut donc venir de la connaissance préalable du système S708, et non des seules 61 agrégations. Elles ne sont disponibles historiquement que **tous les 1024 échantillons**. Une projection last-available vers 256 est causalement possible, mais ne recrée pas une extraction native 256.

## Modifications

Rejouer les 80 fichiers signés S800 avec leur tenseur 49×23 S795 et leur horloge 1024, sans synthétiser de nouvelles probabilités. Évaluer deux LogisticRegression privées, 23 statistiques moyennes contre 61 agrégats identiques au code S810, entraînées à partir du même tenseur et des labels 256. Comparer aussi une sélection sur inner-CAL seulement.

## Protocole

80 GuitarSet FIT80, 5 compositions de 16 enregistrements, 48 TRAIN/16 inner-CAL/16 OUTER. Score original MIDI256 H0 attendu 142213/249411. Pour chaque fin MIDI256, choisir uniquement l'index de la dernière trame S795 disponible: decision_ends[i] <= metric_ends[j]. Avant le premier 1024, remplir d'une donnée manquante neutre sans information future. Déclarer le nombre de trames sous cadence d'observation 1024 et ne pas les présenter comme acoustiques natives 256. Rééchantillonner la vérité des annotations GuitarSet sur 256 uniquement pour labels et métriques. Aucun CAL32/HELD/TEST.

## Audits obligatoires

Signatures des 80 sources S800 et des comptes H0, monotonie des temps, absence de lookahead, emprise du postérieur S708 et des notes source, non-utilisation de labels au moment de la prédiction. 6 autres axes inchangés, source LIVE inchangée. Tout score S823 est **non promouvable** car les features héritées violent l'exigence d'indépendance de Active-K. Pas de preuve de latence micro/CPU ni de généralisation hors FIT80.

## Statut

Expérience de diagnostic préparée; score S823 non mesuré avant exécution CI. Aucun gain revendiqué.
