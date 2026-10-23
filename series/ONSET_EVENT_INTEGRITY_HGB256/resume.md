# Onset — intégrité des événements et expert appris PCM256

Date : 23 octobre 2026. Branche : `codex/note-onset-event-integrity-hgb256`.
Parent : `22b1e61045eeff5ca7ab5b2f7ba0989ce4aba116`.

## Demande actuelle

Continuer à améliorer Onset. Contrainte : 256 nouveaux échantillons à
44 100 Hz par décision, sans fenêtre d'attente, confirmation supplémentaire,
lookahead ou antidatage. L'historique audio passé reste disponible.
Les cinq autres branches ne sont pas modifiées.

## Audit préalable

Les runs 39775357133 et 39905578488 ont terminé leurs replays réels ; leurs
assertions scientifiques ont rejeté les variantes. Référence comparable :
contrôle69, TP 8127 / FP 4519 / FN 3140, F1 67,971396 %.

Un défaut reproductible a été trouvé dans `FNRecoverySelector.select` :
`CausalSelection.step` transmet aussi des `note_off`. L'ancien arbitre ne
filtrait pas leur type avant d'enregistrer le pitch dans `seen` et d'avancer
`last_added`. Une fin de note pouvait donc absorber la réattaque du même
bloc, ou inhiber une attaque ultérieure. L'évaluation d'entraînement recevait
seulement des propositions d'attaque : ce défaut créait une différence entre
l'arbitrage sur événements et le flux audio réel.

## Hypothèses fixées avant le nouveau test

1. Exclure les fins de notes de cet arbitre corrige la causalité de son état.
   Cela peut récupérer des TP **et** des FP ; aucun gain F1 n'est présumé.
2. Des arbres boostés peu profonds peuvent mieux discriminer les erreurs que
   le MLP69 avec exactement les mêmes observations causales. Deux capacités
   sont fixées : 7 ou 15 feuilles, 160 itérations, taux .05, L2 10, minimum
   25 lignes par feuille, sans early stopping. Ce sont de vrais modèles
   entraînés, pas des seuils manuels ni des réseaux neuronaux.
3. Choisir modèle et seuil par appariement exact sur CAL, avec les nouvelles
   attaques récupérées présentes, doit éviter l'approximation qui soustrayait
   simplement les anciennes étiquettes TP/FP. L'expert ne modifie jamais les
   générateurs de candidats : ce réappariement est reproductible sans futur.

## Protocole

- Mêmes 80 WAV signés, 306536 frontières PCM256, vérité terrain originale.
- Premier passage audio : anciens et nouveaux arbitres exécutés en parallèle
  sur les mêmes observations. Les sorties historiques sont reproduites
  exactement : S613, ErrorPrediction, FNRecovery et contrôle69.
- Pour chaque composition OUTER : trois autres compositions TRAIN et une
  CAL. Dix HGB entraînés, modèle et seuil figés avant le calcul OUTER.
- CAL compare le contrôle69 et les deux HGB, avec seuils 0..1 par pas .02.
  On maximise le F1 Onset ; pas de contrainte arbitraire de perte de TP.
  Les égalités conservent la solution la plus simple déjà retenue.
- Deuxième passage audio complet : la tête sélectionnée prédit les 49
  hauteurs à chaque nouveau bloc de 256 échantillons. Vérification des
  événements exacts, pas seulement des compteurs, contre les prédictions
  scellées avant ce replay.
- Les annotations ne sont consultées qu'après l'inférence audio de la piste.
  Identité des WAV, annotations et checkpoints contrôlée par SHA256.

## Limites

FIT80 est un banc de recherche déjà exploré, avec chevauchement historique
d'entraînement des enseignants S613/S701. Une hausse sur ce banc ne prouve
pas la généralisation externe. Le temps de décision de 5,805 ms n'est pas la
latence matérielle. Les mesures du replay instrumenté incluent l'audit et
les comparateurs ; elles ne certifient pas un système microphone/MIDI.

## Reproduction

Workflow : `.github/workflows/onset-event-integrity-hgb256.yml`.
Commande : `python -m scripts.onset_event_integrity256 --root DONNEES --out RESULTATS --phase all --workers 3`.
Le dossier de sortie doit être neuf. Rapports, poids et événements complets
sont conservés même si la nouvelle hypothèse n'améliore pas le F1.
