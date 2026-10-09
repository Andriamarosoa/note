# V27.3 — Série 35 : raccordement du catalogue historique à l'ordonnanceur dynamique (H9 exclu)

**Contrat fixé avant la première mesure.** Demande : « intégrer les autres têtes, sauf H9 ». H9 / YourMT3+ n'est **ni tête exécutable, ni entrée de routage** ; YourMT3+ peut rester une métrique d'évaluation externe uniquement.

## 18 modules connectés, plus A/B historique

1. Prédicteurs acoustiques H1 spectral, H2 cycle de vie, H3 harmoniques, H4 fondamentales, H5 sources ; H6 flux logarithmique, H7 morphologie d'attaque réellement reconstruite à partir de 42 trames ×49 candidats, H10 transport/dissipation énergétique (descripteurs physiques, **pas** un solveur Navier–Stokes). **H8 pitch-shift** : contrat d'exécution avec contrôle complet des identifiants natifs et provenance de vrais fichiers WAV transformés ; tant que ces probabilités ne sont pas disponibles pour les 59 309 cas hors-fold, l'adaptateur reste **masqué**. Ne pas prétendre qu'il a voté.
2. C23, C32, C34, C43 : les quatre adaptateurs de corrections K2↔K3↔K4, éligibles uniquement quand **le K courant** est la classe source, fondés sur les probabilités acoustiques H1/H3 observables. Ils n'introduisent aucune classe vraie dans le choix.
3. F_keep2, F_keep3, F_keep4, F_keep_any : les protections KEEP restaurent **la classe S18 d'origine** si l'état courant a été modifié (et ne réattribuent pas le courant à lui-même). Les trois premières exigent la classe initiale K2/K3/K4 ; la quatrième accepte toutes les classes de départ.
4. E12 : apprentissage natif du spécialiste K1/K2 sur les seules étiquettes K1 et K2 du **train**, mais inférence à partir du signal. Il est distinct des vieux correcteurs K2↔K3/K4.
5. Action AB_stateful_S29 : possibilité de reprendre la proposition véritablement obtenue par le réseau dynamique A/B S29. Le routeur choisit les têtes spécialisées **avant ou après** cette proposition selon l'état ; cette première itération n'entraîne pas encore de nouveau les paramètres de A/B après l'intervention d'une tête historique.

## Prévention de fuite et métriques

- 4 folds natifs [0,1,2,4] (fold3 exclu). Pour le fold évalué, entraîner chaque spécialiste sur les autres trois folds.
- Pour entraîner le routeur, **dans ces trois folds d'entraînement**, produire des probabilités des experts par trois folds intérieurs croisés, chacun entraîné uniquement sur les deux autres folds d'entraînement. À aucune étape un poids de spécialiste utilisé sur le fold externe ne doit voir ses étiquettes.
- Deux états d'entraînement du sélecteur : le parent S18 et une proposition alternative H3 ; ajuster séparément les estimations correction/régression par tête en provenance de ces états hors-fold.
- L'ordonnanceur recalcule le K courant et la valeur attendue de chaque action après **chaque intervention**, trois actions au maximum, sélection A/B et KEEP possibles. Une tête ne peut pas être utilisée deux fois pour un même événement dans cette version. Trois pénalités λ=1,2,4 et trois seuils 0,0.02,0.05, soit neuf politiques annoncées avant résultat.
- Mesurer K0–K6 global, poly K2–K6, fold, corrections, régressions, neutres, chemins et nombre de têtes effectivement sélectionnées ; conserver toutes les décisions et 144 modèles spécialistes OOF (4× [outer + 3 inner] ×9) et les modèles de route.
- L'optimisation **ne calcule pas encore les têtes paresseusement** : les probabilités de tous les modèles disponibles sont pré-calculées avant la décision. Ne pas annoncer d'économie d'appels calculés, contrairement à S27/S29. H8 non exécutée tant que le corpus d'entrées OOF réel manque.
- Les chiffres sur le corpus déjà souvent examiné sont exploratoires ; S18 (82.9183 % global / 40.5958 % poly) reste intacte. Ne pas promouvoir une politique qui casse une seule ancienne bonne sélection, ni sans validation sur compositions réellement nouvelles.

### Provenance
Les têtes H1–H5 étaient auparavant entraînées sur le vocabulaire poly K2–K6 ; conserver ce support structurel dans le routeur. H6/H7/H10 sont de **nouveaux adaptateurs de caractéristiques effectivement calculées** : ce n'est pas une reproduction bit-à-bit de vieux checkpoints ou de l'ancien traitement pitch-shift. H9 exclu.
