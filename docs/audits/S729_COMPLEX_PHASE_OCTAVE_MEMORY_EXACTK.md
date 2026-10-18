# S729 — Phase temporelle des harmoniques partagées, indépendante des amplitudes

## Pourquoi cette boucle diffère de S725 et S727

Quand E2 (82,4 Hz) est déjà active, sa deuxième harmonique (~164,8 Hz)
coïncide avec la fondamentale E3. Les simples valeurs spectrales
(amplitude, énergie de l'harmonique, log-amplitude) ne suffisent pas
nécessairement à décider si une nouvelle corde E3 est excitée.
L'addition physique du signal complexe peut produire un nouveau
comportement des **phases**, même si les pics spectraux se superposent.

Les études sur la séparation des partiels communs par **phase et
trajectoire temporelle** (par ex. Bronson & Depalle, ICASSP 2014,
doi:10.1109/ICASSP.2014.6855053) motivent cette comparaison. Il
existe une limite physique pour les cas de recouvrement total, et ce
travail n'affirme pas résoudre cette limite.

## 288D inchangées + 32D nouvelles (compatibles LIVE causal)

- Utilisation des cosinus cohérence de phase du S706 (24 canaux :
  3 FFT × 8 harmoniques) et des 24 amplitudes logarithmiques ;
- **8D** : discontinuité de phase, modulée par une estimation de
  fiabilité relative à l'amplitude (ne pas considérer une phase
  aléatoire dans une bande quasi silencieuse comme Onset) ;
- **8D** : discontinuité de phase × apparition d'énergie partielle ;
- **8D** : discontinuité de phase × contraste spectral local, pour
  caractériser une excitation plutôt qu'une résonance persistante ;
- **3D** : contraste de rupture de phase pairs/impairs de la note
  plus grave à une octave, conditionné par son Sustain passé ;
- **3D** : contraste des harmoniques multiples de trois à
  +19 demi-tons ;
- **2D** : propriété du parent selon Sustain native EMA observée.

Le signal n'utilise que les frames déjà reçus, et **aucune
annotation JAMS** à l'inférence. La tête neuronale Exact-K K0..K6,
ses interactions avec les 49 têtes de hauteurs, le selector K-guidé
et le NoteOff S707 original restent inchangés.

## Contrôle causal au même budget

Deux entraînements indépendants à partir du **même S708 signé**,
même architecture 320D, même seed, FIT 80, lr, batches et
1 puis 2 époques :
1. `no_phase` : 32 nouvelles entrées remplacées par leur moyenne
   FIT (standardisées à zéro) ;
2. `with_phase` : véritables preuves supplémentaires de phase.

Tous les checkpoints sont soumis au **même décodeur** et aux six
métriques originales de GuitarSet CAL32. La référence S708 n'est
promue automatiquement ni réécrite ; aucune régression sur
Onset/Exact-K global/actif/Offset/full note/frame pitch et aucune
ancienne fin de note correcte ou attaque longue perdue tolérée.
HELD et DEV (déjà exposés) servent uniquement au diagnostic
**après** figement CAL. TEST reste fermé.

[GitHub Actions S729](https://github.com/Andriamarosoa/note/actions/runs/39125385098).
Distinguer la réussite des tests synthétiques (phase pairs E2
perturbée vs phase uniformly perturbée) d'un gain vérifié sur un
vrai instrument : seule l'évaluation de `report.json` établira
ou rejettera cette hypothèse.

## Interprétation prévue des boucles précédentes

- S726 : 791 vrais chevauchements d'octave sur FIT, 151 CAL,
  467 HELD, 23 DEV; beaucoup plus informatif que 1 E2→E3 CAL.
- S727 : pénalité **supervisée FIT** pour faux Onset sur note
  enfant +12/+19 lorsque parent Sustain, sans pénaliser une vraie
  nouvelle E3. Comparaison à poids de 1, 1.8 et 2.6.
- S728 : rejouer les événements réels et compter le rappel des
  151 attaques d'octave et les fausses notes enfants sur CAL.
- S729 : vérifier si la **phase** apporte une information
  mesurable complémentaire aux résidus d'amplitude.

Ne pas mélanger les scores entre lots et ne pas annoncer un résultat
supérieur à 80% non exécuté.
