# S770–S772 — Verdict vérifié (19 octobre 2026)

## Protocole, provenance, limites

- GitHub Actions S770: [39245797266](https://github.com/Andriamarosoa/note/actions/runs/39245797266), deux tentatives sur **le même commit** et les mêmes 80 FIT/32 CAL GuitarSet.
- S771 post-hoc: [39249829288](https://github.com/Andriamarosoa/note/actions/runs/39249829288), **SUCCESS** (65 variantes).
- S772 audit inter-runs: [39250123642](https://github.com/Andriamarosoa/note/actions/runs/39250123642), **SUCCESS** (65 variantes).
- CAL32 a été consulté à de nombreuses reprises. C'est un jeu de diagnostic, PAS un test indépendant ou vierge.
- Aucun nouveau NoteOn/NoteOff MIDI n'est promu ; aucun changement S708/S707 ni LIVE. Les variantes S770 sont ONSET-ONLY SHADOW.

## S770 — deux tentatives du même code

| Mesure | Tentative 1 | Tentative 2 |
|---|---:|---:|
| Run statut | FAIL (parité stricte) | SUCCESS |
| Modèle/combinaison FIT choisie | (qualité 0.3, **sans nouvel Onset**) | identique |
| CAL Onset TP | 2887 | 2885 |
| CAL Onset FP | 791 | 790 |
| CAL Onset FN | 1196 | 1198 |
| CAL Onset F1 | 74.397629% | **74.374839%** |
| Exact-K drift contre S708 signé | +1 trame / 126304 | **0** |
| Active-K drift contre S708 signé | +1 trame / 100243 | **0** |
| Les six scores S708 exactement reproduits | non | **oui** |

S708 signé Onset F1: **74.263862%** (2913 TP, 849 FP, 1170 FN).

S770 tentative 2 fait +0.110977 point Onset F1 en omettant des attaques natives, mais
ce n'est pas un gain des six scores sur un flux MIDI final, et la condition de non-perte
de véritables notes n'est pas vérifiée. Pas de promotion.

## S771 — résultats des 65 boucles

Il y a exactement 65 variantes : 5 seuils qualité native ×
(1 absence d'ajout + 4 seuils naissance × 3 seuils montée spectrale).

Dans **5/5** familles de seuils natifs, même la meilleure variante avec
de nouvelles attaques fait BAISSER le F1 relativement à la même qualité
native sans nouveaux Onsets.

Pour la meilleure qualité native q=0.3 (première tentative) :

- Sans nouveaux Onsets : 2887 TP, 791 FP, 1196 FN, F1 74.3976%.
- Avec meilleur ajout (seuil naissance 0.9, montée 0.1) :
  2931 TP, 881 FP, 1152 FN, F1 74.2495%.
- Gain conditionnel : **44 TP en plus / 90 FP en plus**, précision
  incrémentale 44 / 134 = **32.84%**.
- Une précision incrémentale **strictement supérieure à F1_base/2,
  soit environ 37.20%**, aurait été nécessaire pour un gain positif de F1.
  Il faut donc augmenter la séparation acoustique vraie/fausse attaque,
  pas prolonger les sweeps des mêmes seuils.
- À nombre de FP constant : **363 TP supplémentaires** nécessaires pour
  dépasser 80% (sur les 4083 références).
- À nombre de TP constant : **544 FP à supprimer** pour dépasser 80%.

Ces bornes supposent que tous les autres comptes restent constants ; elles
n'impliquent aucune capacité réelle de récupérer les attaques.

## S772 — défaut de reproductibilité identifié, cause non démontrée

Comparaison byte-for-byte des JSON des deux tentatives (mêmes composants,
données, paramétrages fixés et composition FIT) :

- **64/65 résultats FIT OOF** diffèrent numériquement.
- **43/65 résultats CAL** diffèrent numériquement.
- Le choix FIT reste exactement le même : q=0.3, aucun birth.
- Le score CAL Onset F1 du meilleur choix varie de **0.02279 point**.
- Variation maximale de score F1 CAL parmi les 65 variantes :
  **0.24073 point**.
- La parité S708 a échoué en tentative 1 et réussi sans changement
  fonctionnel en tentative 2.

Cela **ne prouve pas** encore que le seul problème soit la non-détermination
numérique PyTorch ; il faut discriminer variation des caractéristiques en entrée,
prédictions réseau, events MIDI et appariements. Ne PAS relâcher la tolérance
de parité pour faire passer une CI rouge.

## Boucle suivante recommandée (sans réexploiter CAL pour sélectionner)

1. **Audit reproduit par piste, avant annotations** : empreintes SHA des features,
   vecteurs postérieurs, stream MIDI et traces de décision avec marges au seuil ;
   répéter même WAV et checkpoint ≥3 fois en un seul runner, puis dans des runners
   distincts. Journaliser premier échantillon différent pour attribuer la dérive.
2. **Nouveau générateur de propositions** : apprendre des attaques manquées
   sur FIT par hauteur MIDI/fragment, en discriminant fondamentales vs octaves,
   interférences entre cordes, répétitions et spectres transitoires. Les 1073 attaques
   sans proposition mesurées par S767 dans l'ancien générateur ne sont pas réparées
   par un simple arbitre.
3. **Sélecteur dynamique par fragment et hauteur**, formé sur les propositions
   hors-composition FIT (cross-fitting rigoureux), avec apprentissage de
   qualité événements et négatifs difficiles. Vérifier sa précision incrémentale
   avant de construire la paire NoteOn/NoteOff.
4. **Vérification finale** : un seul flux MIDI causal, six scores S704 simultanés,
   conservation d'identité des notes, puis une validation indépendante
   après gel du modèle sur de nouvelles compositions/instruments. Pas de
   distillation ni promotion par seul score ONSET SHADOW.

**État : S708 reste l'officiel.** Aucun des six objectifs >80% simultanés n'est démontré.
