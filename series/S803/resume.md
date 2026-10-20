---
serie: S803
statut: en_cours
type: experience
reference: S708
branche: codex/note-s803-causal-current-frame-gates
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S803 — têtes indépendantes et garde-fous causaux par fragment/hauteur

## Objectif

Dépasser à terme 80 % sur chacune des six métriques sans aucune régression par composition ni globalement. Première étape S803 : confronter trois hypothèses de filtrage d'attaques aux vrais 80 morceaux FIT80 OOF et quantifier les discriminants TP/FP de S802 sans exploiter les données CAL32, HELD ou TEST. Conserver la référence officielle S708, les cinq têtes gelées indépendantes et les choix dynamiques par frame/pitch.

## Hypothèse

La hausse des fausses attaques sur SS1 et Jazz1 pourrait dépendre d'une évidence d'attaque faible, d'un parent harmonique concurrent ou d'un désaccord entre têtes transitoires et générales. Il n'existe aucune preuve qu'un seuil particulier sépare suffisamment les deux classes ; trois seuils pré-déclarés sont des expériences falsifiables, pas des optimisations ni des scores annoncés.

## Modifications

Nouveau module de recherche isolé `scripts/s803_causal_frame_replay.py` et six contrôles `scripts/s803_six_role_audits.py`. Les trois politiques shadow sont `rise_attack` (on >= .20 et on_rise >= .12, issue du routeur adaptive), `parent_conflict` (harmonic_parent <= .02 ou on_rise >= .20, issue du consensus) et `agreement_attack` (general_p >= .96, transient_p >= .93 et on_rise >= .10, issue de l'adaptive). Toutes les quantités sont disponibles à l'instant de décision. Aucune nouvelle tête n'est imposée sans validation, et aucune des cinq têtes existantes n'est fusionnée. Les cinq modèles, les paramètres S795 et le décodeur restent gelés. Aucun fichier de production LIVE, aucune attente de NoteOff, aucun lookahead ou retiming.

## Protocole

Recharger les artefacts originaux GuitarSet S800 #39352880035, S801 #39354733640 et S802 #39359182007, avec vérification SHA256 des 80 traces et parité stricte des comptes S795. ROC AUC par caractéristique et composition sur attaques réelles S802 (vérité *a posteriori*) ; incertitude par bootstrap de pistes. Aucun seuil n'est entraîné ni sélectionné sur la vérité. Rejouer chaque nouvelle variante avec les 80 vrais événements MIDI256 et recalculer les six métriques plus comptes bruts, comparaison S708 FIT source et chaque composition. Veto FAIL pour toute promotion si une seule caractéristique régresse. Toute variation FIT reste exploratoire ; aucune validation CAL32 nouvelle n'est déclenchée. Profil de latence micro physique non mesuré.

## Audits obligatoires

Six jobs distincts et parallèles : Données (FIT80, parité sources et scellage CAL), Acoustique (on/off et parent harmonique/ROC), Architecture (cinq têtes et routage fragment/hauteur), Apprentissage (pas de fuite de vérité ni recherche de seuil), Métriques (six ratios, comptes bruts, veto par composition) et LIVE (aucune altération des événements natifs, absence de buffers futurs). Le conseiller réel NVIDIA NIM est convoqué en parallèle des six audits après l'expérience, par workflow réutilisable. Consensus 7/7 obligatoire pour déclarer le protocole audité, sans pouvoir de promotion.

## Statut

**En cours.** Les métriques S803 sont non mesurées jusqu'au run GitHub FIT80 vérifié et à la présence de ses artefacts. Aucun gain annoncé. Les scripts et le workflow de cette branche ne touchent ni main, ni S708, ni le LIVE actif.
