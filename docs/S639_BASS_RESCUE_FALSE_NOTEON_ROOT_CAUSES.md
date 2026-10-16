# S639 — AUDIT CAUSAL DES FAUX NOTE-ON DE BASS RESCUE ×2

**Verdict : F1 en régression causé par la transformation de preuves spectrales
ambiguës en nouveaux NoteOn, sans contrôle natif de nouveauté/attaque et avec
K natif non contraignant par défaut. Les indices octaves contribuent, mais
ne suffisent PAS à expliquer l'essentiel.**

## Preuves et scope

- 12 enregistrements réels GuitarSet DEV, compositions BN3/Funk3/Rock1,
  annotations originales. Aucune source TEST; DEV avait été étudié.
- GitHub Actions réussi :
  https://github.com/Andriamarosoa/note/actions/runs/38938455675
- Artefact `S639_REAL_GUITARSET_BASS_FP_FORENSIC/report.json`.
- Checkpoint natif gelé et mêmes événements référence S614 :
  **TP998 FP585 FN582 F1 63,104647%**.
- S639 reconstruit les prédictions au moment où le vrai audio est entendu;
  l'annotation n'est consultée que dans la boucle de scoring post-hoc.
- Toutes les expériences ici restent exploratoires et non promues.

## Comptage apparié

Bass ×2, seuil 0,80 :
- **TP1044 FP834 FN536 F1 60,3817235%** ;
- **+46 TP nets** (47 vrais événements récupérés, 1 perdu), **+249 FP nets**.
- 400 occasions de relever les probabilités par rescue : ces occasions
  ne sont PAS toutes des NoteOn.
- Parmi les erreurs nouvelles, **243** fausses attaques n'ont pas de
  correspondant déjà faux en S614 de même pitch à ±80ms.
  Les 6 autres du delta FP net relèvent de l'appariement/changements
  de prédictions et ne sont PAS classées comme entièrement nouvelles.
  Ce n'est pas une partition causale certaine des 249.

## Classes primaires des 243 nouveaux faux événements diagnostiqués

Ces classes sont exclusives par ordre des heuristiques et les relations
harmoniques désignent un **soupçon**, non une preuve audio FFT.

| Classe descriptive | Nombre | Part |
|---|---:|---:|
| Fuite depuis l'attaque d'une autre hauteur, ±80ms | 98 | 40,3% |
| Réattaque d'une note déjà annotée comme tenue | 37 | 15,2% |
| Doublon de NoteOn dans la fenêtre d'une attaque vraie | 35 | 14,4% |
| Coïncidence avec une hauteur vraie à l'octave | 34 | 14,0% |
| Coïncidence avec rapport harmonique (+/−19 ou 24 demi-tons) | 30 | 12,3% |
| Autre sustain actif inexpliqué | 4 | 1,6% |
| En dehors d'une note vraie active | 5 | 2,1% |
| **Total** | **243** | **100 %** |

- Les erreurs proches d'une autre attaque forment le groupe majeur :
  **98/243**. Lors de la frappe d'un accord, plusieurs partiels se
  renforcent ; la vue ×2 ne prouve pas qu'une **fondamentale nouvelle**
  a physiquement démarré.
- Réattaques + doublons : **72/243** ; le décodage met `onset` à son
  seuil sans nouvelle preuve temporelle native. 58/243 suivent un
  même-pitch NoteOn précédent à moins de 120ms et 80/243 à moins
  de 300ms, statistiques non exclusives.
- Relations octave + autres intervalles harmoniques : **64/243**.
  Le soupçon est basé sur un intervalle entre pitch prédit et hauteurs
  annotées tenues, pas sur une reconstruction physique validée.

## Preuve que la fonction de fusion court-circuite l'attaque normale

Le code `scripts/s637_guitarset_three_view_groundtruth.py::corrected` :
- exige score transposé ≥0,80 et sustain transposé ≥0,50,
- impose `result[mask,0]=state.config.onset` et
  `result[mask,1]=max(native_sustain, shifted_sustain)`,
- ne demande **pas** que le score d'attaque native monte.
En cas de réactivation du sélecteur, l'état de sustain est aussi dominé
par le flux transformé.

Mesures aux 243 nouveaux faux événements :
- **242/243 (99,6%)** ont l'attaque native sous 0,60 ;
- **191/243 (78,6%)** ont le sustain natif sous 0,50 ;
- **166/243 (68,3%)** ont même le sustain natif sous 0,25 ;
- **112/243 (46,1%)** ont une prédiction modale K native ≤ nombre de
  notes déjà actives, mais le décodeur officiel ne les utilise pas
  pour bloquer l'attaque.

Ces chiffres expliquent pourquoi une confiance élevée en rescue ne se
traduit pas par une bonne précision MIDI : une fenêtre modifiée peut
soutenir la note sans corroboration native.

## Preuve de la désactivation de K et comparaison des garde-fous

Dans le `Selection` gelé :
- `count_confidence = 1.1`.
- `CausalSelection.step` n'applique le budget K que si
  `k.max() >= count_confidence`. Une probabilité ≤1 ne peut atteindre 1,1.
  K est donc **prédit mais non appliqué comme plafond natif** (max 6).

| Configuration | TP | FP | F1 | Écart de points vs S614 |
|---|---:|---:|---:|---:|
| S614 (K inactif) | 998 | 585 | 63,10 % | 0 |
| Bass 0,80 (K inactif) | 1044 | 834 | 60,38 % | −2,72 |
| Bass 0,80 + veto octaves | 1032 | 782 | 60,81 % | −2,29 |
| Bass 0,80 + veto K ≥35 % *sur nouvelles propositions* | 1034 | 765 | 61,20 % | −1,90 |
| S614 avec K global 0,35 | 966 | 552 | 62,36 % | −0,74 |
| Bass 0,80 avec K global 0,35 | 1003 | 700 | 61,10 % | −2,00 |

Le veto K local à 35% conserve +36 TP nets au prix de +180 FP.
Un veto K global protège certaines fausses notes, mais fait perdre
**32 TP réels** à S614. **Il serait incorrect de simplement fixer
count_confidence de 1.1 à 0.35 dans le modèle officiel.**

Le veto octave local réduit les faux positifs du rescue mais perd
également 12 gains vrais : l'octave explique une partie des erreurs,
pas leur totalité.

## Pourquoi particulièrement dans les notes superposées

S638 établit sur 1147 vrais NoteOn avec superposition au départ :
- S614 : 697 détectés, rappel **60,77%**.
- Bass0,80 : 740 détectés, rappel **64,52%**.
  Il y a 44 annotations récupérées et 1 perdue en superposition.
- Pour K3+ : 37 nouveaux vrais retrouvés, 1 précédemment vrai perdu.
- Faux NoteOn durant passages polyphoniques : **416 → 603** (+187).
- Globalement F1 **63,10 → 60,38%**.

La transposition révèle donc réellement certaines fondamentales
masquées d'un accord, mais multiplie aussi les **pitchs parasites
corrélés avec la même attaque d'accord**.

## Recommandation de conception — non encore mesurée comme gain

1. Ne pas promouvoir ce rescue ni forcer le K global.
2. Garder les 3 vues comme producteurs de preuves/candidats,
   pas comme prédicteurs NoteOn autoritaires.
3. Ajouter une tête de sélection **entraînée uniquement sur TRAIN**,
   conditionnée sur : variation de fondamentale réelle au moment courant,
   nouveauté et flux spectral natifs, compatibilité des harmoniques,
   risque d'octave, candidats concurrents à la même attaque, K natif
   probabiliste et état actif/armement/refractory.
4. Différencier **une nouvelle attaque** d'une **note qui continue de
   vibrer**, avec supervision des attaques et faux réarmements.
5. Vérifier la nouvelle tête sur DEV puis GuitarSet/URMP/IDMT TEST scellés
   indépendamment; mesurer TP/FP/FN/F1 séparément par K et par registre.
6. Avant toute promotion LIVE, mesurer CPU, files micro et latence causale.

Le défi quantitatif est sévère : à +46 TP constants, Bass ×2 peut
ajouter au maximum **99 FP** pour battre F1 S614; il en ajoute 249.
Il faudrait éliminer au moins 150 de ces nouveaux faux événements
sans sacrifier les 46 TP.

## Limites du mot « cause »

Les classes octave/harmonique et les catégories d'attaque sont fondées
sur les horaires/hauteurs annotés et les scores au moment de l'émission.
Elles **identifient une proximité très suggestive** et des faiblesses
mécaniques dans le code, mais pas une séparation prouvée de l'énergie
audio en fondamentale et partiels. Une FFT/STFT locale ou un audit
d'énergie concurrentielle serait nécessaire pour prouver la cause
spectrale d'un faux événement individuel.
