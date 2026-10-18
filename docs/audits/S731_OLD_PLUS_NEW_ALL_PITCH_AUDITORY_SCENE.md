## Correction de causalité pendant la validation

Le NoteOff S707 peut porter un `decision_sample` antérieur
à son émission effective (`available_sample`). L'audit source
ne doit jamais supprimer une source dans son historique
*avant* `available_sample`.
Le moteur S731 traite désormais les événements **uniquement
au premier bloc effectivement reçu après leur disponibilité**.
Le contrôle vérifie explicitement le cas où un NoteOff est
daté à 2048 échantillons mais n'est connu qu'à 5120 :
la note reste active pour les décisions intermédiaires.
[Tests corrigés : SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/39128765445).
[Validation complète après correction](https://github.com/Andriamarosoa/note/actions/runs/39128785724).
Ne pas utiliser les résultats d'un run démarré avant ce correctif
pour conclure à une véritable causalité LIVE.

---

# S731 — « Comme l'oreille » : ancien son + nouvelle source, pour toutes les notes

## Ce qui change réellement après l'échec S720–S730

S725 et S729 ont tenté d'injecter de nouvelles valeurs dans le modèle
TemporalHeads initialisé depuis S708 ; les gains CAL n'ont pas survécu à
la validation conjointe Onset/Exact-K/Offset/full-note.
S730 a essayé d'ajouter des événements spécialisés tout en conservant
l'ancien flux S708 : les fausses attaques et la métrique Offset se
dégradent lorsque davantage de notes sont émises. Ces résultats
excluent l'idée qu'une simple montée de score Onset ou un seuil K
suffira à reproduire la séparation perceptive.

**S731 n'est pas une autre extension 288→320D du réseau.**
C'est un *mécanisme causal de reconstruction perceptive de la scène*.
À chaque décision on compare explicitement :
- **H0 — ancien son :** le nouveau pic est une harmonique ou
  une réexcitation générale d'une ou plusieurs sources déjà actives ;
- **H1 — nouvelle source :** le nouveau pic a une évolution harmonique
  indépendante impossible à expliquer par la décroissance commune
  et les fluctuations du volume des sources anciennes.

Exact-K est **maintenu** comme conditionnement appris des probabilités
Onset/Sustain/Release S708 et comme indice `P(K > nb_notes_actives)`
dans le score H1 ; la source acoustique reste indispensable : un
K=2 isolé ne crée aucune nouvelle note.

## Couverture universelle

Pour **les 49 hauteurs MIDI 40–88**, le graphe physique
`src/causal_note/s731_auditory_scene.py:harmonic_ownership`
calcule toutes les correspondances harmoniques compatibles entre
un pitch candidat et n'importe quelle source MIDI 40–88 active.
Une fréquence candidate est attribuable à l'ancienne source si
elle coïncide à moins de 18 cents avec sa partielle 1..32
(les bandes effectivement mesurées fournissent seulement
les huit premières harmoniques de chaque pitch, sur trois fenêtres).
Ceci inclut les octaves, quintes et autres rapports entiers
sans règles spéciales E2/E3.

**Attention :** l'existence d'une attribution harmonique
possible ne prouve pas une attribution réelle. De même,
l'absence d'harmonique mesurée dans les huit premières
partielles d'une source active ne prouve pas qu'une autre
note a été jouée : le graphe couvre donc jusqu'à 32.

## Mémoire acoustique causale

État persistant par pitch : derniers 8 partiels × 3 FFT,
tendance de décroissance bornée, notes réellement émises
sur la chaîne S708 avant la décision courante.
Les observations ne sont pas la vérité terrain et ne
contiennent pas de signaux futurs.

Pour chaque note candidate :
1. prédit le spectre de l'instant courant à partir de la
   dernière observation et de la pente passée ;
2. extrait les innovations positives non prédites ;
3. cherche les notes **déjà actives** qui peuvent expliquer
   les mêmes pics via leurs harmoniques ;
4. mesure les changements communs de ces parents sur
   leurs huit premières partielles (un gain global ou une
   nouvelle attaque de l'ancienne corde est un indice H0) ;
5. retranche seulement cette partie explicable du nouveau
   signal, sans effacer une hausse propre à une source
   indépendante dont les fréquences se superposent ;
6. exige un soutien cohérent de plus d'un partiel ;
7. pondère H1 par Exact-K ; ne modifie que la tête Onset,
   jamais Sustain, Release, K ni la politique Offset S707.

Ce prototype ne sépare pas physiquement des pistes audio
individuelles et ne reconstitue pas encore les phases
complexes de chaque source ; il teste une hypothèse
de séparation *sur les événements* et non une imitation
intégrale de l'oreille ou du cerveau.

## Preuves techniques

[Tests S731 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/39128305645)

- hausse uniforme des partiels d'une ancienne note :
  annulation de la nouveauté partagée quand l'ancien
  parent explique la hausse ;
- nouvelle excitation harmonique indépendante :
  innovation partagée positive même si le pic de
  fondamentale était déjà une harmonique d'une autre
  note ; testé sur l'octave et la quinte de toutes les
  hauteurs valides, sans branche `if E2` ;
- pas de source inventée par une tête K qui reste sur K=1 ;
- le Sustain et le Release ne sont jamais modifiés ;
- futures observations réécrites n'affectent jamais
  une preuve ni une source active du passé ;
- reprise des tests S730 et du contrat des six métriques.

## Validation sur les vraies prises, sans changement de définition des scores

[Workflow S731](https://github.com/Andriamarosoa/note/actions/runs/39128488318)
rejoue le même audio GuitarSet et le **S708/S707 signé original**
pour toute la grille pré-déclarée de 9 valeurs
`gain(H1)∈{0.15,0.35,0.65}` ×
`explain(H0)∈{0,0.05,0.10}`.
Ce ne sont pas 9 réseaux entraînés : les mêmes probabilités,
et uniquement les nouvelles décisions d'explication de la scène,
sont confrontées de manière indépendante.

Chaque candidature produit un seul flux causal de NoteOn/NoteOff
complet, évalué sur CAL32 pour les six métriques,
plus les **151 vraies attaques d'octave** et **67 attaques à
+19 demi-tons** déjà documentées (rappel correct / notes
supérieures inventées) et les pertes d'anciens vrais
Onsets longs/Offsets. HELD32 et DEV12, historiquement exposés,
ne sont rejoués qu'après le gel du candidat CAL.
TEST reste fermé. Un gain d'une seule métrique ne
justifie **aucune promotion LIVE**.

### Verdict
Ne pas annoncer de gain S731 avant l'exécution réelle
et l'archivage de `report.json`. Même un algorithme
inspiré de l'analyse de scène auditive ne peut résoudre
des sources physiquement non identifiables à partir du
signal mono, et ses hypothèses de décroissance / gain
doivent être validées selon les instruments et conditions
d'enregistrement.
