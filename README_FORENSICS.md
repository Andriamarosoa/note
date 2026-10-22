# Onset — preuve harmonique causale PCM256 (observation seulement)

Branche : `codex/note-onset-harmonic-forensics-256`. Cette expérience est issue de `codex/note-onset-s701-trained-256` et **ne remplace aucun poids**, n'ajoute aucun délai ni veto et ne modifie **aucune des cinq autres caractéristiques ni le MIDI officiel**.

## Objet

Le dépôt a déjà démontré avec S684 que beaucoup d'attaques manquées ont une probabilité Onset trop faible et que beaucoup de fausses attaques sont proches de notes harmoniques. S785 a montré qu'un vérificateur d'attaque **par événement** peut être utile, mais son modèle est entraîné sur S708 à 1024 : on ne peut pas employer ce modèle directement. S830 a montré comment ajouter des comparaisons entre notes graves et harmoniques dans un espace de caractéristiques PCM256, sans sortie provenant d'une autre caractéristique.

L'expérience présente capture, à l'instant **exact** où S613 ou S701 émet une attaque, les indices acoustiques courants `[49,126]`, le postérieur S613 et le score S701, puis ajoute :

- force fondamentale du candidat et des parents à −12/−19/−24/−28 demi-tons ;
- force des voisins ±1, rapports d'énergies et variation temporelle disponible **avant** la décision ;
- probabilités Onset/Sustain/Release et compte K de S613 comme **signal auxiliaire interne**, pas comme dépendance au décodeur Exact-K ni comme écriture MIDI ;
- indicateur `harmonic_parent_stronger` purement diagnostique. Aucune attaque n'est filtrée.

Le module `src/onset256/harmonic_evidence.py` n'a accès **ni aux annotations, ni au fichier MIDI vérité, ni aux buffers futurs**. `Stream(audit_events=True)` renvoie les mêmes attaques qu'avant mais joint à chacune un contexte calculé sur la trame courante. Les tests exigent la parité de toutes les sorties avec/sans audit, la même cadence (256 échantillons ; 5,805 ms), l'invariance au découpage audio et les contrôles de frontière.

Le replay scientifique `scripts/onset_harmonic_forensics256.py` mesure en post-traitement de chaque morceau les TP, FP et FN par hauteur, registre, texture et indice harmonique. L'assignation des annotations à un événement se fait **uniquement après la fin complète de l'inférence PCM**. Il est crucial de distinguer un événement FP proche d'une note voisine dans la vérité d'une vraie preuve physique de cause : ces observations constituent une **corrélation descriptive**, pas un diagnostic causal certifié.

## Contrats de reproductibilité

- WAV GuitarSet originaux et manifeste S800 signés SHA256, 80/80 enregistrements, cinq compositions ×16, 306 536 vrais appels PCM256 par modèle.
- Checkpoints S613 (SHA256 `7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`) et S701 (SHA256 `cc72cf81f477c6b734e253df5fb04d1789ab707dc160f06bdc5cf9ffdff2fb0f`) déjà entraînés à 256.
- Scores inchangés exigés : S613 **8099 TP / 4600 FP / 3168 FN** ; S701 shadow **8113 TP / 4603 FP / 3154 FN**. Si un score change, le run échoue plutôt que de conclure à une amélioration artificielle.
- FIT80 a été vu pendant les entraînements S613/S701 ; **aucune promotion, aucun résultat de généralisation indépendante** n'est possible à partir de cet audit.
- Fichiers CI : `report.json` (agrégats par texture/registre/force du parent harmonique) et `events.jsonl` (chaque événement avec indices strictement antérieurs/courants, puis étiquette TP/FP/FN ajoutée post-inférence).

## Critère d'avancement suivant

Les groupes FP « parent harmonique plus fort » seront comparés aux groupes TP. Si leur distribution diffère réellement, l'architecture d'un futur vérificateur inspiré de S785 utilisera ces indices **avec une tête nouvellement entraînée sur TRAIN+CAL indépendants de l'OUTER**, plutôt que d'imposer arbitrairement un veto. Une attaque réelle faible avec un parent harmonique fort doit rester détectable si les données d'entraînement le justifient. Les réattaques à même hauteur utiliseront les annotations de début réelles ; ne pas employer les pseudo-étiquettes `occupancy & ~previous` de S830.

Le pipeline LIVE reste strictement causal et sans ajout de `hold` ou `lookahead`. Aucun p95 CPU ou latence matériel iRig n'est revendiqué avant un benchmark mesuré.
