# Onset pré-décision : réseau de risque neuronal natif 256

Statut : **en attente de validation scientifique**.

Méthode : cinq vrais `StandardScaler + MLPClassifier(64,32)` sur événements Onset S613 avec 63 caractéristiques causales et labels TP/FP, split 3 TRAIN / 1 CAL / 1 OUTER compositions par pli, plus un modèle groupé non promu. Prédiction pour 49 notes **avant** chaque appel au sélecteur. Hop acoustique, hop d'entraînement et hop de décision **256**.

Baselines originales signées FIT80 : S613 TP8099 / FP4600 / FN3168 = 67,5874 % F1 ; S701 SHADOW TP8113 / FP4603 / FN3154 = 67,6563 % F1 ; S708 historique poids1024 TP8528 / FP3489 / FN2739 = 73,2520 % F1.

Le workflow doit produire (1) modèle neuronal réellement appris et signé SHA256, (2) AUC probabilités de risque sur chaque OUTER, (3) F1 candidat événementiel OOF, (4) F1 **véritable décodeur causal** avec MLP appelé avant chaque décision sur 80 pistes WAV originales, (5) bilan apparié corrections/régressions et (6) pannes éventuelles. Les F1 nouveaux ne sont PAS encore mesurés. Tout score de régression est rejeté (aucune promotion et MIDI public inchangé).

Limites : les têtes originales S613/S701 ont connu des morceaux de FIT80 lors de l'entraînement. OOF du nouveau risque seul ne prouve pas généralisation du pipeline complet. Aucun modèle entraîné1024/S708 importé.
