# S598–S605 — Boucles de correction, apprentissage et test réservé

Protocole du 15 octobre 2026, après S597. Objectif : au moins 80 % de F1
micro NoteOn, hauteur exacte MIDI40–88 et tolérance ±80 ms, sur l'intégralité
d'une nouvelle cohorte réservée. Les scores à ±30/50 ms, par morceau et les
régressions individuelles sont également publiés. Un gain sur les données
déjà exposées ne vaut pas validation indépendante.

## Défaut démontré et premières boucles

S597 a trouvé 180 notes vraies de Basic Pitch direct absentes de S585, dont
176 sans raccordement protégé à V88. La première ablation conserve les
horodatages de toutes les notes BP, puis ajoute les candidats S585 selon
trois seuils d'attaque (0,7/0,8/0,9) et trois distances d'exclusion vis-à-vis
de BP (80/100/160 ms). Ces neuf variantes sont des mesures de développement
sur les cinq URMP et les deux piano S591 déjà exposés. Même un score de
80 % sur ces deux piano ne suffirait pas à conclure que l'objectif est atteint.

Les boucles suivantes utilisent les sorties neurales de Basic Pitch à leur
horloge native : activités par note, attaques et contours. Elles examinent
les notes courtes, la compétition entre hauteurs voisines, les répétitions,
et une sélection apprise de candidats. Le réseau Basic Pitch 0.4.0 reste
figé ; une éventuelle nouvelle tête apprend uniquement à partir du groupe
d'entraînement déclaré ci-dessous. Aucun texte de partition, instrument
annoncé dans le nom de fichier ou label de la cible ne sert d'entrée.

## Réservation avant nouvel audio ou annotation

Voir [le manifeste complet](v273_s599_urmp_train_test_metadata_only.json).
Même miroir et même révision immuable qu'en S592. Le sélecteur reprend le
SHA256 du chemin et les catégories cordes/bois/cuivres/mixte/cordes, en
excluant les cinq titres exposés dans S597. Il réserve :

| Test | Instruments |
|---|---|
| 44 K515 | 2 violons, 2 altos, violoncelle selon le nom du fichier préparé |
| 40 Miserere | 2 flûtes, hautbois, clarinette, basson |
| 33 Elise | 2 trompettes, cor, trombone |
| 18 Nocturne | Violon, flûte, trompette |
| 02 Sonata | 2 violons |

Les arrangements 17 Nocturne et 41 Miserere sont aussi exclus de
l'entraînement. **37 enregistrements, 22 titres** constituent le développement
et l'entraînement ; les **cinq titres du test restent absents**, toutes
instrumentations confondues. Les cinq anciennes sources S597 appartiennent
désormais au développement. Aucun remplacement d'un morceau difficile
après observation des résultats. Tous les extraits restent fixés à 5–29 s.

Les folds d'apprentissage et de sélection doivent être groupés par **titre
musical**, jamais par frame, note ou simple enregistrement : deux arrangements
de la même œuvre restent dans le même fold. Les métriques hors fold guident
les seuils et choix de décodeur. Ces métriques servent au développement et
ne sont pas présentées comme le résultat final indépendant. Les variantes
essayées, leurs paramètres et leurs résultats doivent rester traçables.

## Gel et verdict

Le candidat final, ses poids éventuels, paramètres et règle de sélection
sont enregistrés et empreintés avant toute inférence sur les cinq WAV test.
Les prédictions complètes sont ensuite figées avant l'accès aux annotations
Notes_ de ces cinq sources. Les partitions Sco_ ne sont jamais utilisées.
Le test publie Basic Pitch direct par défaut comme témoin, la variante
globale sélectionnée en développement et le candidat appris retenu, si ce
dernier est justifié en développement. La première évaluation de la cohorte
est conservée, même si une nouvelle boucle est nécessaire par la suite.

Le résultat principal compte toutes les notes admissibles, tous les FP/FN,
fusionne seulement les unissons de même hauteur au même instant exact, et
publie les notes hors plage. Le nouveau décodeur devra expliquer ses notes
récupérées et perdues. L'objectif de 80 % concerne le F1 global du test, pas
le meilleur morceau ni un registre choisi après coup.

La mesure reste sur audio préparé 16 kHz, avec cinq courts extraits et des
annotations humaines. Ce n'est pas une validation de l'identité instrumentale,
de NoteOff, de la latence ou du Live. Les scripts et poids historiques S80 et
Exact-K ne sont pas modifiés ; une promotion de leur référence demanderait
toujours leur validation indépendante spécifique.
