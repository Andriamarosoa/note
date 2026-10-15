# S606–S609 — Transfert de S601 figé sur GuitarSet, IDMT et MAESTRO

Protocole enregistré avant l'inférence de transfert et avant l'ouverture des annotations de cette évaluation. **Aucun résultat sur ces trois cohortes n'est encore revendiqué.** Les 84,91 % précédents concernent uniquement les cinq extraits URMP réservés de S605.

| Corpus | Sélection sans audio ni annotations | Extrait |
|---|---|---|
| GuitarSet microphone | 60 enregistrements : six guitaristes × cinq styles × solo/accompagnement ; plus petit SHA256 du nom parmi les six fichiers de chaque cellule | À partir de 5 s, jusqu'à 24 s ou la fin du fichier |
| IDMT-SMT-GUITAR v2, Dataset3 | Les cinq enregistrements ; `pathetique_mono` et `pathetique_poly` restent deux arrangements du même morceau | À partir de 0 s, jusqu'à 24 s ou la fin du fichier |
| MAESTRO v3 | 12 enregistrements, un par compositeur à crédit simple ; split officiel test, durée 29–300 s, plus petit SHA256 du nom, exclusion des 35 sources des huit manifestes antérieurs explicitement listés | 5–29 s |

Les courtes prises IDMT sont évaluées dès leur début pour ne pas supprimer l'essentiel de l'enregistrement. Une durée restante inférieure à deux secondes provoque un échec explicite, sans remplacement du fichier. La sélection et les identités des fichiers sont dans [le manifeste S606](v273_s606_cross_corpus_preregistration.json).

## Modèles et mesure figés

La tête S601 reste exactement celle de S602 : SHA256 `f925fa0487abe299b7ef1978d2163bfcfbea8fade8e43a105e22bbf7636d6cfb`, 168 variables, seuil 0,5, espacement minimal des prédictions de même hauteur 100 ms, décalage temporel nul. Les contrôles sont Basic Pitch 0.4.0 par défaut et le décodeur global S600 précédemment choisi sur URMP. Ni apprentissage, ni ajustement par instrument, ni recherche de seuil ne sont permis.

Mesure principale : F1 micro NoteOn, hauteur MIDI exacte 40–88 et appariement un à un à ±80 ms. Les résultats à ±30 et ±50 ms, précision, rappel, TP/FP/FN, registres, fichiers et strates sont aussi conservés. Les annotations hors tessiture sont dénombrées séparément. Seuls les doublons d'unisson au temps exactement identique sont fusionnés ; les réattaques restent des événements distincts. Les JAMS sont arrondis au demi-ton MIDI le plus proche, sans déplacement de leur temps d'attaque ; les MIDI MAESTRO sont lus aux vrais NoteOn avec leurs changements de tempo, sans regroupement d'accords ni extension par pédale.

## Séparation de l'inférence et du score

Le workflow [S606–S609](../.github/workflows/v273-s606-s609-frozen-cross-corpus.yml) produit trois archives de prédictions. Le calcul du score ne commence qu'après la fin des trois tâches audio et vérifie leurs empreintes. Les archives contiennent les sorties Basic Pitch, les candidats, leurs probabilités et les trois politiques figées. Aucun fichier audio n'est publié. L'accès partiel à IDMT vérifie les CRC des membres extraits et conserve leurs SHA256 ; le MD5 de l'archive entière est publié par Zenodo mais n'est pas recalculé lors de cet accès partiel.

Le rapport S609 doit publier tous les corpus, y compris les régressions : chaque vraie note gagnée ou perdue face à Basic Pitch, les faux positifs ajoutés/supprimés, les faux négatifs et les réattaques de moins de 100 ms. Deux algorithmes d'appariement vérifient indépendamment les comptes TP/FP/FN de chaque fichier aux trois tolérances.

## Portée de l'indépendance

La tête S601 n'a appris que sur URMP. GuitarSet et MAESTRO font cependant partie des corpus de préentraînement de Basic Pitch : **ce test ne démontre pas une indépendance vis-à-vis du préentraînement du réseau de base**. IDMT Dataset3 et certains fichiers GuitarSet ont déjà servi aux anciens modèles du projet. Les exclusions MAESTRO couvrent les huit manifestes cités, sans prétendre couvrir toute l'histoire du dépôt. Les morceaux courts de MAESTRO constituent un échantillon défini par le protocole, pas l'ensemble de son test officiel.

Cette validation porte sur le microphone GuitarSet, les cinq prises IDMT et les douze prises de piano choisies. Elle ne certifie pas d'autres guitares, microphones ou pièces acoustiques. NoteOff, Exact-K complet, causalité et Live restent hors de cette mesure. Aucune promotion de S80 ou modification du Live n'est autorisée par ce score.
