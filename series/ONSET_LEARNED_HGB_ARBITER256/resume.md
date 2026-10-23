# Onset — arbitre HGB/acoustique par fragment PCM256

23 octobre 2026. Branche codex/note-onset-learned-hgb-arbiter256.

## Hypothèse préenregistrée
L'audit révèle 825 événements HGB autrefois TP et désormais perdus ; 802 en accompagnement, 821 sur Rock3/SS1. Une réactivation globale HGB donne 809 anciennes pertes récupérées mais 1510 FP de plus, F1 69,163123 %, et est rejetée.

Hypothèse testable : un modèle supervisé par proposition HGB conserve les décisions qui ajoutent des TP sans restaurer les FP et fait progresser le F1 au-delà de 69,206204 %.

## Architecture
- Référence HGB et acoustique directe 195D gelées.
- Nouveau modèle HistGradientBoostingClassifier 207D : 195D acoustiques + 12 valeurs de contexte strictement causal (score direct, historique q -1/-3 et max des sept hops passés, delta, rang parmi les 49 hauteurs, densité des notes, voisins fréquentiels, âge dernière attaque directe/HGB, pitch normalisé).
- Ajout sélectif HGB sans veto de la tête acoustique ; exclusion des doublons par le passé exclusivement, aucun bloc futur.
- Grille 256 échantillons à 44,1 kHz ; aucun audio futur et aucune attente ajoutée.
- Aucune modification des autres cinq caractéristiques et du MIDI officiel.

## Protocole
1. Reproduire strictement 80 WAV originaux / 306536 hops, avec reference HGB TP8050 FP4231 FN3217 et direct TP7585 FP3068 FN3682.
2. Étiqueter par appariement un-à-un les candidats HGB et retrancher les annotations reconnues par le direct. Jamais de label à l'inférence.
3. Pour chaque OUTER (composition), utiliser trois compositions TRAIN, la quatrième CAL ; probabilités acoustiques OOF pour toute proposition.
4. Entraîner cinq vrais arbitres, mesurer les politiques (seuil 0,05–0,95, gap 1/7/14) sur CAL seulement. Le candidat peut conserver la tête directe seule.
5. Figer chaque modèle et politique avant OUTER ; mesurer TP, FP, FN, F1, récupérations et pertes annotation par annotation.
6. Replay audio complet des 80 WAV, égalité des événements Live et caches, vérification causalité PCM256.
7. Rejeter tout candidat dont le F1 OUTER total ne dépasse pas 69,206204 % ; ne jamais promouvoir le résultat sans tests et validation externe indépendante.

Les nouveaux scores restent non mesurés. Le FIT80 a un overlap historique ; ce n'est pas une validation externe.

## Commandes
    python -m scripts.onset_learned_hgb_arbiter256 --experiment "$ONSET_EXPERIMENT_DIR" --out "$GATE_DIR" --phase prepare
    python -m scripts.onset_learned_hgb_arbiter256 --experiment "$ONSET_EXPERIMENT_DIR" --out "$GATE_DIR" --phase train
    python -m scripts.onset_learned_hgb_arbiter256 --experiment "$ONSET_EXPERIMENT_DIR" --out "$GATE_DIR" --root "$ONSET_DATA_DIR" --phase replay
