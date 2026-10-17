# S642 — Note isolée vs note superposée à une octave (GuitarSet réel)

## Question

Est-ce que E2 seul se distingue d'un accord E2 + E3, alors que
**E3=164,8 Hz est déjà l'harmonique n°2 d'E2=82,4 Hz** ?

Une raie spectrale à 164,8 Hz ne peut pas répondre à cette question
à elle seule. Pour une note E2 isolée, il y a déjà les harmoniques
164,8, 247,2, 329,6, etc. Pour E2+E3, les fondamentales
82,4 et 164,8 et les harmoniques E3 à 329,6, 494,4, etc. se
superposent aux harmoniques paires d'E2.

## Audit de disponibilité (GuitarSet TRAIN)

- Toutes les 36 pistes TRAIN GuitarSet ont été inspectées via les
  annotations originales, sans lire DEV ni TEST.
- E2 (MIDI40) : **71 occurrences**, mais **0 attaque isolée** sous
  critère : aucune autre hauteur active dans les ~50ms avant
  et les 120ms suivant le début, note >=250ms.
- E3 (MIDI52) : **188 occurrences** dont **9 attaques isolées**.
- E4 (MIDI64) : **14 attaques isolées**.
- Par conséquent, prétendre tester E2 seul contre E2+E3 avec
  **GuitarSet TRAIN** serait trompeur. Le protocole S642 utilise le
  **couple octave analogue E3 (164,8 Hz) / E4 (329,6 Hz)**.

## Méthode réelle : huit paires contrôlées

Sources individuelles extraites de fichiers micro GuitarSet originaux
avec vérité terrain note_midi, pistes et compositions TRAIN uniquement.
Pour chaque paire :
1. exactement le même extrait réel E3 seul ;
2. E4 seul provenant d'une autre composition GuitarSet ;
3. E3 avec E4 lancé **simultanément**, somme PCM de deux vraies prises
   de guitare à niveau RMS aligné (gain équivalent) ;
4. E3 puis E4 **120 ms après**, somme PCM de la même paire.

Les accords 3 et 4 sont des **mixages numériques construits à partir
de vraies prises instrumentales**, pas des performances de guitariste
réellement jouées en même temps. La tête S614 originale + K reste gelée.
Le rapport mesure les NoteOn dans une tolérance de 80ms.

- CI réussie : https://github.com/Andriamarosoa/note/actions/runs/38997795464
- Artefact `S642_REAL_GUITAR_NOTE_VS_OCTAVE_COMPARISON/report.json`.
- Branch `codex/s642-e2-e3-true-octave-vs-harmonic`.
- **Ce sont des prises TRAIN** : ces extraits font partie des pistes
  auxquelles NOTE original a eu accès pendant son développement.
  Cette expérience diagnostique n'est PAS une évaluation de
  généralisation indépendante, ni une estimation de F1 global.

## Résultats — attaques reconnues à ±80ms

| Stimulus | Attaques vraies E3 détectées | E4 détectées | NoteOn vrais détectés |
|---|---:|---:|---:|
| E3 isolé | **8/8** | Non applicable | 8/8 |
| E4 isolé | Non applicable | **8/8** | 8/8 |
| E3+E4 ensemble, même temps | **6/8** | **8/8** | **14/16** |
| E3 puis E4 +120ms | **8/8** | **5/8** | **13/16** |

Ces fractions reflètent exactement **8 couples de courts extraits
contrôlés**, avec certains audios d'origine répétés dans des couples.
Elles ne sont pas des pourcentages généralisables.

Pourquoi note grave manquée quand les deux attaques sont synchrones ?
La réponse E4 peut dominer le mélange et la tête de sélection
peut ne pas émettre E3. Il s'agit d'une interprétation liée aux
prédictions, non d'une preuve causale universelle.

## Mesures du spectre et des têtes dans la fenêtre 40–120ms après attaque

Comparaison des mêmes résolutions physiques natives sur GuitarSet :

| Cas | Énergie log f0 E4 (329,6Hz) | Score d'attaque E4 par NOTE |
|---|---:|---:|
| E3 seul | 2,2526 | 0,0555 |
| E4 seul | 3,0392 | 0,2993 |
| E3+E4 simultanés | 2,9058 | 0,3121 |
| E3+E4 décalés de 120ms — avant arrivée E4 | 2,0348 | 0,0536 |
| E3+E4 décalés — après arrivée E4 | 2,8998 | 0,3913 |

La bande 329,6Hz existe déjà sur E3 seul à cause de son harmonique 2,
mais **son attaque native ne ressemble PAS à une nouvelle note E4** :
0,0555 contre 0,3121 quand E4 est réellement superposé.

Lorsque E4 démarre plus tard, son score d'attaque passe en moyenne
de 0,0536 avant à 0,3913 après son arrivée — **preuve d'une
nouveauté détectable en temps causal**, même quand les hauteurs sont
à une octave.

Pour E2 seul vs E2/E3, le même principe s'applique au couple
82,4Hz/164,8Hz mais **les chiffres mesurés ci-dessus concernent
E3/E4 et NE SONT PAS des mesures E2/E3**.

## Attention aux faux positifs

Les petits extraits (1,65s) ne sont pas des studios mononote purs :
on sélectionne des attaques isolées dans les annotations mais
d'autres notes peuvent précéder/suivre, laisser une queue acoustique
ou dépasser la vérité simplifiée des deux stimuli construits.
Par conséquent, les nombreux faux NoteOn « hors E3/E4 » du JSON
ne doivent PAS être interprétés comme un F1 GuitarSet valide.
La conclusion ne porte que sur la correspondance des attaques
ciblées E3 et E4 à ±80ms.

## Consequence architecture NOTE

**Pour distinguer E2 d'E2+E3 :**
1. Présence E2 f0 ~82,4Hz doit être vérifiée physiquement ;
2. E3 ~164,8Hz peut être l'harmonique 2 d'E2 : ne suffit jamais seul ;
3. Un onset indépendant de la bande E3 et un changement de ses
   harmoniques supérieures (329,6/494,4) peuvent soutenir E3 vrai ;
4. Si les deux notes commencent simultanément, l'onset commun
   ne sépare pas leurs sources. Ajouter **une tête entraînée**
   E2-only vs E2+E3 sur véritables exemples multi-instruments
   / prises indépendantes, avec gabarits harmoniques et contraste
   de résidus ;
5. Rejouer K1/K2 et l'émission NoteOn au même sample, puis mesurer
   sur accords vrais jamais vus. Un K2 n'est pas une preuve de
   hauteurs correctes sans la sélection.
6. Ne pas promouvoir ce test : il est diagnostic, construit et
   basé sur TRAIN.

## État

- Code fonctionnel et testé en CI, huit paires réelles.
- Aucune modification du NOTE officiel ni du système LIVE.
- Le défi pour généraliser E2/E3 est de disposer de **vrais E2
  isolés et vrais duos E2/E3**, et de tester des contextes variés
  sans contaminer les partitions scellées.
