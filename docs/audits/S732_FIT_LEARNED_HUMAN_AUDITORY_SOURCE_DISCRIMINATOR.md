## Verdict mesuré S732 — rejet

[Run GitHub 39130284723 : SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/39130284723), 18 tests, deux modèles HGB entraînés seulement sur FIT80, 359 834 observations d'entraînement, dont 29 030 positifs Onset. **S708 seul sélectionné**. Aucune promotion LIVE.

Sur CAL32, les variantes `control` et `scene` sans pénalité (gains 0,20 / 0,45 / 0,75) produisent exactement les mêmes événements et les mêmes six métriques que S708 :

| Score CAL | S708 | S732 sélectionné |
|---|---:|---:|
| Onset F1 | 74,2639% | 74,2639% |
| Exact-K | 62,0455% | 62,0455% |
| Exact-K actif | 56,2762% | 56,2762% |
| Offset | 77,3292% | 77,3292% |
| F1 note complète | 57,1319% | 57,1319% |
| F1 hauteur/trame | 76,4626% | 76,4626% |

Le modèle `scene` avec `penalty=0,08` a modifié quelques décisions : Onset 74,2908%, K 62,0099%, K actif 56,2034%, Offset 77,3513%, full-note 57,1684%, frame-pitch 76,4040%. **Régression K et frame-pitch.** Sur les 151 vrais chevauchements d'octave de CAL32 : S708 = 99 vrais et 43 faux enfants ; S732 `scene` pénalisé = 97 vrais et 40 faux enfants. Deux vrais perdus pour trois faux supprimés. Seul le modèle S708 original est acceptable ; HELD32 et DEV12 reproduisent donc exclusivement les résultats historiques S708.

### Cause structurelle de l'échec

Le nouveau HGB prédit `P(onset)`, mais `apply_evidence` n'augmente la tête Onset S708 que lorsque le HGB est **plus confiant que S708** ; ce supplément est ensuite multiplié par `P(K>nb_actifs)` du réseau S708. Les tests complets révèlent que ces corrections positives n'ont produit aucun NoteOn supplémentaire, dans les trois amplitudes testées. Les corrections négatives changent certains événements et créent le compromis faux-vrais. Cela ne signifie PAS qu'aucune information de scène n'existe : **ce couplage après-coup n'exploite pas correctement les cas que la tête S708 et son Exact-K ratent déjà**.

La prochaine rupture à éprouver doit être une architecture entraînée **conjointement** à expliquer le signal par plusieurs sources indépendantes, avec mécanisme de compétition/reconstruction au coeur de la sélection (ou tête indépendante sur événements candidats), plutôt qu'une probabilité ajoutée à la sortie de S708 et verrouillée par son K inchangé. Il faut mesurer le résultat sur des splits réellement inconnus et toutes les combinaisons de notes, pas multiplier les réglages CAL.

---

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
