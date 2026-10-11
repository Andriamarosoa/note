# S74 — Audit avant changement de référence (11 octobre 2026)

**Décision : changement officiel BLOCKED ; candidat expérimental S72 conservé.** Les résultats S73 (49,05 % poly) n'ont pas de prédictions individuelles archivées, et ses régressions globales sont fortes : **ne pas le substituer à S72**.

## Mesure exacte : S72 poly vs référence officielle freeze_local_combo

Sources alignées sur 59 309 identifiants natifs :
- Référence PR16 run `37837235723`, artefact `v273-open-summary`, champ `baseline_K`.
- Candidat S72 run **[38155427761](https://github.com/Andriamarosoa/note/actions/runs/38155427761)**, artefact `v273-s72-temporal-risk`, champ `s72_poly_poly_guard_K`.
- 4 archives de 190 enregistrements GuitarSet, folds 0/1/2/4, SHA-256 acoustiques vérifiés ; les rapports de feature extraction déclarent `fold3_used=false` et `player05_used=false`.

| Mesure | freeze_local_combo | S72 | Delta |
|---|---:|---:|---:|
| Exact-K global (59 309 fragments) | 48 454 / 59 309 = **81,70 %** | 49 238 / 59 309 = **83,02 %** | **+784** |
| Exact-K poly (7 385 vrais K2–K6) | 2 530 / 7 385 = **34,26 %** | 3 202 / 7 385 = **43,36 %** | **+672** |
| Corrections vs incumbent | — | **2 558** | |
| Régressions vs incumbent | — | **1 774** | |
| Régressions sur les vrais accords | — | **708** | |
| Sous-comptages tous K | 6 539 | 5 340 | **−1 199** |
| Surcomptages tous K | 4 316 | 4 731 | **+415** |

La différence poly par fold 0/1/2/4 est **+195/+176/+162/+139** ; toutes positives. Gain global par fold **+221/+166/+132/+265**. Les 190 enregistrements : **133 gains, 20 égalités, 37 pertes** (sur observations historiquement exposées).

Par vrai K, nombre correct incumbent → S72 :
- K0 : 38 035→37 934 (**−101**)
- K1 : 7 889→8 102 (+213)
- K2 : 1 183→1 730 (**+547**)
- K3 : 915→1 043 (+128)
- K4 : 417→389 (**−28**)
- K5 : 15→39 (+24)
- K6 : 0→1 (+1)

## Boucle S74 exécutée avant cet audit

24 filtres de risque acoustique/votes/combinaison, seuils inner-OOF sur trois folds, et 36 routeurs par transition old-K→new-K. **Aucun ne conserve le niveau de polyphonie S72 en diminuant substantiellement les régressions.** Le meilleur routeur avec contrainte poly non négative améliore seulement 1 accord et 1 décision globale (3 groupes autorisés, 39 veto). Il est rejeté comme nouveau candidat de référence.

**Conflit identifiable :** K1→K2 est nécessaire à la récupération des accords (242 corrections poly sur cette transition), mais 383 bonnes décisions d'autres classes sont perdues pour 242 corrections nouvelles. Supprimer cette transition améliore le global au détriment du poly.

## Contrôle de promotion (bloquant)

Une amélioration sur les folds utilisés pendant l'exploration n'est **pas une validation indépendante**. Les prédictions archivées des modèles S72 ne couvrent que les folds **0,1,2,4**. Le **fold 3 / joueur 05**, explicitement absent des 4 rapports acoustiques, est une piste de test verrouillée mais ne possède **aucune prédiction comparative native S72 vs incumbent** dans ces artefacts.

Pour changer officiellement la référence :
1. Geler code, poids et seuils de S72 (et ses producteurs PR16/S70/S71) **avant** de consulter les étiquettes de fold3.
2. Produire les prédictions incidentes sur fold3 avec des producteurs dont les entraînements excluent entièrement fold3, vérifier provenance des prédites et enregistrements.
3. Auditer simultanément global K0–K6, poly K2–K6, corrections/régressions, et vérifier un gain positif sur les deux domaines ; détailler K0/K4 et les pertes par enregistrement.
4. En cas de succès, demander validation de la PR pour changer le pointeur officiel ; ne jamais mélanger ce test avec un nouveau réglage des seuils.

**Statut officiel conservé** : `freeze_local_combo`. **Candidat de promotion gelé pour audit** : `S72 poly_poly_guard`. **S73** reste seulement exploratoire. Aucun résultat fold3 ou indépendance n'est inventé.