# V27.3 — Trajectoires harmoniques et cycle de vie énergétique

**Date : 2026-10-08. Statut : protocole exploratoire terminé ; référence intacte.**

Les deux comptages binaires de perte d'énergie sur des templates quasi nuls ont
été remplacés par des rapports continus avant le run final, afin de limiter
la sensibilité aux arrondis. Les scores du run final 37739605464 prévalent
sur les premières valeurs exploratoires.

- [Run initial, complet](https://github.com/Andriamarosoa/note/actions/runs/37738998181)
- [Run final après stabilisation, complet](https://github.com/Andriamarosoa/note/actions/runs/37739605464)
- Branche : `codex/v273-failure-clustering`
- Cohorte : **1 800 événements** natifs, 150 K2 + 150 K3 + 150 K4
  dans chacun des folds **0, 1, 2, 4**. Joueur 05 et fold 3 exclus.
- Même échantillonnage déterministe et mêmes splits que
  `v273-energy-transport-k234.yml`.
- Objectif : analyser l'énergie comme un ensemble de trajectoires
  qui naissent, persistent et s'éteignent, **sans** utiliser des
  fréquences de notes extraites des étiquettes.

## Mesures

Construction d'un spectrogramme 4096/256 à 44,1 kHz, 240 ms de contexte
(-80/+160 ms). Dictionnaire fixe de 49 fondamentales candidates MIDI 40–88,
six premiers harmoniques, puis reconstruction non négative à dictionnaire
fixe et lissage non négatif. Les sorties sont des caractéristiques :
état harmonique, variation de sources candidates, flux d'apparition, persistance,
atténuation et continuité.

**ATTENTION :** les activations de templates chevauchants ne sont pas
des notes isolées. Ce n'est pas une inversion acoustique de Navier–Stokes.
L'analyse exploite 160 ms d'audio après l'événement et n'est pas causale.

Les tests synthétiques confirment uniquement que ces observables réagissent
aux apparitions/coupures connues et demeurent finies.

## Résultats K2/K3/K4

Mesures d'un **classifieur diagnostique à trois classes**, mêmes 1 800 cas
équilibrés, entraîné sur trois folds et évalué sur le quatrième.
Ce ne sont PAS les scores du réseau `freeze_local_combo` à sept classes.

| Représentation | Balanced acc. | Macro AUC | Rappel K4 |
|---|---:|---:|---:|
| Énergie statique, 24 bandes | 46.50 % | .6457 | 56.8 % |
| Flux + transport 24 bandes | 47.06 % | .6572 | 55.8 % |
| État harmonique seul | 49.56 % | .6774 | 58.5 % |
| **Naissance + maintien + extinction** | **51.28 %** | **.6895** | **67.0 %** |
| Harmoniques, toutes caractéristiques | 51.67 % | .7061 | 59.8 % |
| Harmoniques désaccordés | 51.83 % | .6996 | 62.2 % |
| **Fondamentales seules** | **50.72 %** | **.7082** | **60.5 %** |
| Ordre temporel brouillé | 49.39 % | .6789 | 59.2 % |
| Flux 24 bandes + cycle de vie harmonique | **52.61 %** | **.7174** | 60.7 % |

### Stabilité inter-folds (balanced accuracy)

| Fold | Énergie statique | Cycle de vie harmonique | Tous traits harmoniques |
|---|---:|---:|---:|
| 0 | 45.11 % | **50.00 %** | 52.89 % |
| 1 | 47.33 % | **51.33 %** | 51.78 % |
| 2 | 49.33 % | **51.78 %** | 49.11 % |
| 4 | 44.22 % | **52.00 %** | 52.89 % |
| Ensemble | 46.50 % | **51.28 %** | 51.67 % |

Le cycle de vie harmonique gagne donc face à l'état statique **dans les
quatre folds**, dont le fold 2. La paire est **316 corrections / 230
régressions**, net +86 / 1 800 exemples par rapport au classifieur statique
du protocole **à trois classes** ; cela **n'est pas** un gain validé de
`freeze_local_combo`.

La combinaison totale avec les anciennes caractéristiques atteint une
macro AUC de .7174 et une balanced accuracy de **52.61 %**, meilleure que
les 51.67 % du modèle harmonique seul. Sur le fold 2, elle obtient
**54.89 %** (contre 49.33 % pour le contrôle statique).
Elle réalise 347 corrections et 237 régressions (net +110) par rapport
au contrôle à trois classes. Cette combinaison a été évaluée sur les
mêmes folds historiquement exposés : **aucune validation indépendante**.

## Interprétation honnête

1. **Information de durée de vie :** les statistiques de
   naissance/persistance/extinction sont utiles et leur avantage sur
   l'énergie statique apparaît dans les quatre folds testés.
2. **Harmoniques pas identifiés de manière causale :** fondamentales seules
   donnent déjà 50.72 %. Des partiels désaccordés donnent presque les
   mêmes performances que les partiels vrais. L'importance de la structure
   harmonique exacte et la séparation des véritables sources musicales
   restent non démontrées.
3. **Effet du temps :** le brouillage temporel réduit la balanced accuracy
   de 51.67 % à 49.39 %, mais ce contrôle modifie aussi certaines frontières
   du mouvement. Il ne constitue pas à lui seul une expérience causale.
4. **Pas de validation indépendante :** les compositions de GuitarSet des
   folds internes ont déjà été examinées. La distribution K2/K3/K4 est
   artificiellement équilibrée et diffère du problème complet K0–K6.
5. **Pas de promotion :** `freeze_local_combo` reste inchangé,
   les scores de référence demeurent les scores précédents ; le joueur
   05 et le fold 3 n'ont pas été utilisés pour cette recherche.

## À tester ensuite

- Définir un test synthétique *aveugle* avec combinaisons multiples et
  extinction physique contrôlée : naissance, masquage, sustain et mute
  par note, et décomposition dans des templates concurrents.
- Tester une attribution harmonique temporelle avec contrôle fondamental
  identique, puis mesurer les fausses sources par harmonique partagée.
- Intégrer uniquement les traits de cycle de vie dans une branche
  expérimentale du système global K0–K6, sans toucher au
  `freeze_local_combo` sauvegardé.
- Valider sur nouvelles compositions/enregistrements non utilisés pour
  la sélection des caractéristiques ; ne pas retoucher les paramètres
  sur le joueur 05.

## Code

- `scripts/extract_v273_harmonic_trajectory.py`
- `scripts/summarize_v273_harmonic_trajectory.py`
- `test/test_v273_harmonic_trajectory.py`
- `.github/workflows/v273-harmonic-trajectory-k234.yml`
