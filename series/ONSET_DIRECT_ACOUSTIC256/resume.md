# Onset — apprentissage acoustique direct PCM256

Référence de départ : commit `8ef0b734bee41a41b4868e194091708194a2ef1f`,
F1 **68,3709869203 %**, TP8050 / FP4231 / FN3217 sur GuitarSet FIT80.

## Hypothèse

Les correcteurs 69D ne reçoivent pas directement les 126 mesures acoustiques
natives, et leurs propositions restent dépendantes des seuils Onset/Sustain
et du budget du sélecteur historique. Une nouvelle tête apprend une attaque
par hauteur à chaque bloc, sur **195 entrées = 126 acoustiques + 69 de contexte
causal**. Elle peut proposer une hauteur même si le détecteur historique ne
produit aucun candidat. Ce n'est pas un transfert des poids historiques 1024.

Le sélecteur accepte immédiatement un franchissement de seuil ; réarmement
et exclusion des doublons utilisent uniquement le passé. Aucun événement
en attente, aucune confirmation par une trame future, aucune rétrodatation.
L'état du détecteur historique reste indépendant de la décision de cette tête.

## Protocole figé avant entraînement

- Original : 80 WAV / 306536 blocs de 256 à 44100 Hz, archives et annotations SHA.
- Cinq compositions OUTER ; la suivante dans l'ordre BN1/Funk2/Jazz1/Rock3/SS1
  sert de CAL, les trois autres de TRAIN pour les nouveaux poids.
- Deux cibles : attaque annotée dans les 6 ou 12 blocs précédents. Ce sont des
  fenêtres de supervision, **aucune attente imposée au flux**.
- HGB : 240 itérations, 31 feuilles, 195 entrées ; 10 modèles au total.
- Négatifs : échantillonnage déterministe 2,5 %, ou 20 % lorsque l'ancien score
  Onset dépasse 0,12. Voisinage temporel ambigu exclu des négatifs.
- CAL compare détection directe, union avec la référence, et union avec veto
  acoustique. Seuils, réarmement et exclusion des doublons sont choisis sur le
  F1 événementiel CAL seulement. Garder la référence est une possibilité.
- Modèle et politique sauvegardés avant les prédictions OUTER. Tous les essais
  et leurs régressions sont conservés ; aucun réglage à partir des scores OUTER.
- Nouvelle lecture audio après sélection : parité de toutes les entrées 195D,
  des événements proposés et des événements de la référence 68,370987 %.

La calibration utilise une seule composition par pli ; sa robustesse reste
à examiner. FIT80 et les modèles amont ont déjà servi à des expériences :
ce protocole ne constitue pas une validation externe indépendante.
Priorité aux scores, optimisation de performance différée.

## Reproduction

Après restauration des archives signées dans `$ONSET_DATA_DIR` :

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_EXPERIMENT_DIR" --phase collect --workers 3
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_EXPERIMENT_DIR" --phase samples
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_EXPERIMENT_DIR" --phase train --workers 3
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_EXPERIMENT_DIR" --phase replay --workers 3
```

Le dossier de sortie doit être neuf. Les modèles, données intermédiaires et
rapports ne sont jamais écrasés par une nouvelle exécution.

Cette série concerne uniquement Onset, sans promotion du MIDI ni modification
des autres caractéristiques.

## Seconde boucle : calibration sur quatre compositions

La première calibration choisit la détection directe seule pour le pli Rock3,
à partir de SS1. Le score réservé Rock3 régresse de 68,344519 à 66,183167 % :
482 vraies attaques perdues, 741 faux positifs retirés. Ce choix n'est pas
remplacé après consultation de Rock3 ; l'expérience entière reste archivée.

Une seconde expérience apprend et sélectionne avec quatre compositions internes
par OUTER. Les dix paires de compositions exclues couvrent vingt séparations
TRAIN3/CAL1. Les cinq modèles à fenêtre de supervision 12 blocs déjà appris
sont réutilisables pour cinq paires ; cinq autres modèles sont appris. Chaque
composition de CAL est exclue du modèle qui prédit ses probabilités.

Le seuil, le réarmement et le mode sont choisis sur le F1 micro des quatre
compositions internes réunies. La grille comporte .75/.80/.85/.90/.95/.975/.99,
deux réarmements et deux exclusions des doublons. La référence reste candidate.
Les poids choisis sont ensuite réappris sur les quatre compositions internes
et figés avant OUTER. Seule la supervision à 12 blocs est utilisée : elle avait
gagné les comparaisons CAL disponibles de la première boucle. Aucune attente
de 12 blocs n'est introduite à l'inférence.

```bash
python -m scripts.onset_acoustic_grouped256 --previous "$ONSET_EXPERIMENT_DIR" --out "$ONSET_GROUPED_DIR" --workers 5
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_GROUPED_DIR" --phase replay --workers 3
python -m scripts.onset_acoustic_direct256 --root "$ONSET_DATA_DIR" --out "$ONSET_GROUPED_DIR" --phase audit
```

Le second dossier référence les mêmes caractéristiques audio par liens locaux.
Les rapports identifient les modèles réutilisés, chaque composition exclue et
les SHA des probabilités. FIT80 est un corpus de recherche déjà exposé, y
compris par la première boucle ; cette reprise n'est pas un nouveau test aveugle.
