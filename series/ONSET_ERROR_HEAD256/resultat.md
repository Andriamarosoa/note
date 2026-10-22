# Série Onset — erreur prédite comme vraie tête du modèle PCM256

**Branche :** `codex/note-onset-integrated-error-head-256`. **Entraînement et replay scientifique en cours de validation GitHub.** Aucune promotion.

## Têtes privées Onset
- `main_attack` : S613, poids entraînés à 256.
- `rescue_attack` : S701, poids entraînés à 256, 39 caractéristiques causales.
- `error_prediction` : vrai MLP indépendant, 63→64→32, entraîné sur TP/FP S613 sur les trames originales 256, poids et calibration propres ; il prédit `P(erreur)` **avant la sélection**.

## Critères quantitatifs
- Tout le pipeline est recalculé sur 80 WAV GuitarSet avec 306 536 décisions PCM256.
- `HeadOpinions` calculé avant les sélecteurs, `Doubt` explicite sur chacune des attaques du main, nombre de TP contestés et de FP contestés mesuré après l'inférence.
- Le main reste strictement identique : TP 8099, FP 4600, FN 3168, F1 67,5874 %, régression tolérée **zéro**.
- La sortie de sélection alternative `predecision_intents` doit être scorée séparément ; aucun nouveau F1 n'est inventé avant exécution.
- Comparaison au S708/H0 historique 73,2520 % seulement à titre indicatif, ses poids 1024 sont exclus de cette architecture.
- Contrôles des SHA, des séparations composition-OOF, de la causalité, de la signature PCM et de l'indépendance des cinq autres axes.

## Limites
Les poids S613 et S701 avaient vu des prises FIT80 ; le nouvel OOF du MLP ne suffit pas à établir la généralisation externe du pipeline complet. Les coûts CPU p95/p99 et la latence matérielle n'ont pas encore été mesurés. Il n'y a aucune écriture dans le MIDI officiel ni changements de décodeur des autres axes.
