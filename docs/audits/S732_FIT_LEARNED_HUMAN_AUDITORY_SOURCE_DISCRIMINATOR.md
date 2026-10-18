# S732 — Apprendre à distinguer une vraie nouvelle source d'une ancienne harmonique

## Point de départ factuel

**S731 (run GitHub 39128785724) est terminé : CAL32 encourageant, HELD/DEV non validés, aucune promotion.**

| Métrique CAL GuitarSet | S708 officiel | S731 old+new, gain 0,15 |
|---|---:|---:|
| Onset F1 | 74,2639 % | **74,3264 %** |
| Exact-K | 62,0455 % | **62,3227 %** |
| Exact-K actif | 56,2762 % | **56,6354 %** |
| Offset | 77,3292 % | **77,3805 %** |
| F1 notes complètes | 57,1319 % | **57,2191 %** |
| F1 hauteurs | 76,4626 % | **76,7370 %** |

La règle S731 améliore individuellement les **six scores CAL32**
et conserve les anciennes attaques longues/offsets corrects,
mais échoue en HELD et DEV. Une hausse localement sélectionnée
sur CAL n'est donc pas une preuve de généralisation.

Le diagnostic ciblé CAL sur les 151 vrais nouveaux onsets d'octave
reste **99/151 corrects** (43 faux onsets enfant), donc l'objectif
de séparation de sources harmoniques **n'est pas atteint**.

## Nouvelle hypothèse S732

Au lieu de régler neuf coefficients de S731 sur CAL,
**apprendre sur FIT80** ce qui constitue vraiment une
excitation indépendante. Réseau S708 d'origine strictement
figé : entrée 288D, 49 hauteurs, Exact-K K0..6,
politique selector et modèle NoteOff S707 **inchangés**.

Détecteur de sources distinct `src/causal_note/s732_source_discriminator.py`
utilisant 21 indices par note et instant reçu :
- les prédictions causales Onset/Sustain/Release de S708,
  K room, probabilité polyphonique et entropie K ;
- la source **ancien son** : quels partiels le réseau
  possède déjà et peut expliquer dans le spectre ;
- la source **nouveau son** : innovation positive non expliquée,
  nombre de partiels cohérents et degrés d'explication H0/H1 ;
- phase causale reçue, sensibilité au niveau du son,
  montée du niveau local, contraste spectral et registre ;
- aucune vraie note annotée ne rentre comme indice de prédiction.

### Ablation indispensable

Les **deux modèles HistGradientBoosting** ont exactement
les mêmes paramètres, FIT80, seed et labels ; le contrôle
n'a que les 10 indices acoustiques/neuronaux génériques,
et le modèle scène dispose aussi des indices d'attribution
des fréquences et de phase.

Il ne suffit pas que S732 gagne contre S708 : il faut
que le modèle **avec analyse des sources** apporte
une information supplémentaire au contrôle **sans scène**,
et qu'il tienne face aux données non utilisées pour
l'ajustement du score.

### Protocole

1. Rejouer les 80 sources FIT signées. Construire le
   flux historique S708 et les indices acoustiques
   **avant de lire les annotations** de cette source.
2. Ouvrir les annotations FIT pour entraîner l'estimation
   binaire : « cette note débute réellement maintenant »,
   contre anciens harmoniques / bruits / pas d'attaque.
   Conserver **tous les positifs** ; échantillonner
   aussi les négatifs difficiles, avec correction
   partielle de la probabilité d'échantillonnage.
3. Rejouer les mêmes 32 CAL en aveugle pour chaque
   modèle et 6 combinaisons prédéclarées de force
   de correction Onset et pénalité H0.
4. Ne jamais modifier Sustain, Release, Exact-K, les
   probabilités d'origine S708 ou le moteur S707.
   Les décisions sont causales et datées à la
   disponibilité réelle des événements.
5. Sélection seulement si **les six métriques** CAL
   sont non-régressives, aucun Onset long/Offset correct
   perdu, aucun TP global perdu ; ensuite
   contrôle HELD32/DEV12 historiques.
6. Reporter séparément les 151 attaques d'octave
   et les 67 attaques à +19 demi-tons :
   notes réelles récupérées **et** fausses
   notes enfant, sans sélectionner sur HELD/DEV.

### Limites

Une représentation fréquentielle mono ne permet pas de
séparer mathématiquement tous les signaux lorsqu'ils sont
identiques. Cette approche est une estimation causale
et supervisée de l'indépendance acoustique, pas un
« lecteur mental » des cordes ou un véritable clone
du système auditif humain.

**Aucune promotion** et aucun changement au moteur
microphone LIVE avant validation mesurée, coût CPU
et latence de streaming inclus.

[Workflow de validation S732](https://github.com/Andriamarosoa/note/actions/workflows/note-s732-fit-learned-sources.yml).
