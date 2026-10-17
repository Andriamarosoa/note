# S668–S670 NOTE NoteOff : progrès causal, mais 80 % non atteint

## Périmètre

Référence réelle S614/S630, corpus GuitarSet MIC authentique :
36 extraits TRAIN, 12 DEV, 9 compositions TRAIN et 3 compositions
DEV distinctes. Une seule sortie NoteOn native est utilisée, et
les postérieurs K0–K6 ne sont jamais modifiés.

Précision calculée sur **990 fins observables** parmi les notes
NoteOn correctement appariées, tolérance stricte ±50 ms.
La livraison S630 **batch ONNX ×4 est simulée par arrondi de la
disponibilité au paquet de quatre trames** ; le benchmark ne prouve
pas le débit CPU/microphone réel ni l'absence d'overruns.

| Variante | Sélection TRAIN | DEV NoteOff ±50 ms, batch4 |
|---|---|---:|
| Original S630, attente batch4 | — | 553/990 = 55,86 % |
| S668 étude de fin apprise | later_0.30 | 671/990 = 67,78 % |
| S668 hazard_0.50 observée, **non retenue sur TRAIN** | non | 762/990 = 76,97 % |
| S669 filtre du canal release existant | original | 553/990 = 55,86 % |
| **S670 preuve spectrale + rythme des attaques déjà reçues** | **hazard_0.50** | **764/990 = 77,17 %** |

S670 est le meilleur des candidats sélectionnés sur
**TRAIN tenu par composition**, *avant* l'évaluation DEV. Son
score tenu TRAIN est **1 312/2 271 = 57,77 %**, contre le
décodeur natif 916/2 271 = 40,33 % ; il y a un écart de distribution
important entre le joueur TRAIN et DEV. Les sources GuitarSet sont
historiquement déjà exposées au projet ; ce n'est ni un test
prospectif scellé ni une validation matérielle du microphone.

**80 % sur 990 notes = au moins 792 fins correctes : il en manque 28.**
Il n'y a donc AUCUNE promotion du S670 ni modification du S630 LIVE.

## Méthode

S668 et S670 entraînent un gradient boosting sur des trames par
note précédemment détectée. Les exemples et étiquettes de fin
viennent exclusivement de GuitarSet TRAIN. Lors de l'inférence,
les entrées sont uniquement les FFT/indices NOTE causaux, le
poste d'attaque natif, les distributions K et la mémoire des
têtes au plus jusqu'à la trame courante. S670 ajoute la phase
rythmique estimée à partir des **NoteOn déjà détectés seulement**,
sans mesurer le futur.

La stratégie « hazard_0.50 » émet à la première paire de trames
consécutives dépassant 0,50, sans antidater le temps de sortie.
Aucune annotation n'intervient dans l'inférence des sorties DEV.
Les prévisions ne sont pas encore intégrées au code live.
Les sorties CPU/PortAudio iRig n'ont pas été mesurées.

## Workflows

- S668 : https://github.com/Andriamarosoa/note/actions/runs/39027788880
- S669 : https://github.com/Andriamarosoa/note/actions/runs/39027750017
- S670 : https://github.com/Andriamarosoa/note/actions/runs/39027929083

Autres expériences programmées indépendamment sur branche :
S671 entraînement multi-joueurs, S672 grille de seuil/confirmation.
