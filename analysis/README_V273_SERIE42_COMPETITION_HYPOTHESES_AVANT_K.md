# S42 — correction structurelle : compétition des hypothèses de notes avant le comptage

**Décision de conception fondée sur des audits existants, pas un gain expérimental déclaré.** La question initiale (« insuffisant ») est justifiée : S36b n'améliorait le poly que de 2/7385 événements et les nouvelles boucles S38–S41, bien qu'offrant jusqu'à 1 090 vraies corrections supplémentaires potentielles, détruisent trop de prédictions correctes S18.

## Constat sur les expériences réellement achevées

- S38, run 37880386031 : quatre nouveaux classifieurs K1–K6 sur original 335 et morphologie 825. Meilleur **K2–K6 standalone 37,0481 %**, sous S18 40,5958 %. Jusqu'à 1 086 corrections globales S18 face à 858 régressions pour la variante HGB/poly argmax ; aucune admissible pour remplacer S18.
- S39, run 37880791041 : les quatre têtes S38 sont complémentaires de S18 sur **1 000 accords poly vrais**, leur **oracle non déployable** donne 3 998/7 385 = **54,1368 %** de bonnes réponses. Ces 1 000 n'ont jamais été sélectionnées correctement ensemble par un modèle réel ; toute comparaison à YourMT3+ 54,5430 % reste informative et **non valide comme performance d'un modèle**.
- S40, run 37880762289, indépendamment revérifié run 37880850747 : CNN temporel sur 42×49 séquence + caractéristiques 335, **38,1720 % poly standalone**. **1 090 accords poly supplémentaires** corrects hors S18, mais **1 269 accords poly** corrects S18 incorrects CNN. Meilleure intervention globale 83,4056 % avec 1 360 corrections et 1 071 régressions ; poly 36,2762 %. Aucune promotion.
- S41, run 37881191015, replay indépendant 37881332296 : **19 portes neuronales**, six actions (S18 garder + 4 spécialistes S38 + CNN S40), fit sur autres morceaux du fold avec sources hors fold. Meilleur global **83,1830 %, poly 37,0616 %**, 763 corrections et 606 régressions ; **aucune option sans perte**. H9 exclue des six actions et des features.

### Antériorité déterminante : V17.7

Le rapport [V17.7 post-audit](v177-post-audit.md) établit *sur une autre cohorte (76 768 événements)* que :
- le nombre attendu de véritables naissances était **0,549**, mais la masse d'objets candidats estimée **1,293** (**+135,5 %**), en raison du modèle Bernoulli par hypothèse de candidat ;
- à vrai K3 et K4, le nombre d'hypothèses actives devenait 8,256 et 12,383, respectivement, malgré seulement 3 ou 4 véritables événements ;
- les pénalités négatives par candidat étaient diluées dans les ensembles comptant beaucoup de faux candidats ; à K6 certaines hypothèses surnuméraires recevaient une **pénalité négative nulle** ;
- le choix de l'identité et du temps était presque résolu une fois le bon objet identifié, mais pas **combien de candidats indépendants existaient réellement**.

Ce rapport **ne prouve pas** que les erreurs du routeur S35/S41 ont exactement la même cause : ses représentations et sa cohorte diffèrent. Mais il fournit un défaut de conception démontré à tester directement, au lieu de supposer que toutes les fréquences fortes représentent des notes.

## Architecture suivante à construire, pas à présenter comme déjà entraînée

`original audio / spectre 42×49 → candidats (f0, instant, enveloppe, énergie, persistance, structure harmonique) → graphe de redondance / compétition → groupe d'hypothèses physiques → une présence par groupe → K et identités de notes → réévaluation par ordonnanceur dynamique`

Règles vérifiables :
1. Les candidats proches en fréquence et en temps ou correspondant à des harmoniques/subharmoniques d'une même source **doivent pouvoir se concurrencer** ; ne pas attribuer une Bernoulli indépendante à chaque pic.
2. Toutes les sources **distinctes** (une attaque indépendante, persistance d'énergie distincte, comportement différent lors de l'étouffement) doivent pouvoir survivre ensemble ; une relation de fréquence harmonique n'est pas une preuve suffisante d'une seule note.
3. Utiliser une perte **candidate / note vraie**, construite avec les annotations originales sur les seuls folds d'entraînement, et une perte de cardinalité **après** regroupement. Ne pas diluer l'ensemble des négatifs par le nombre de candidats disponibles.
4. Calculer et conserver le graphe, les groupes, les propositions refusées, le score et chaque décision ; tests d'invariance contre la duplication des hypothèses, ablation de l'attaque, notes voisines, harmonic aliasing et étouffement.
5. Évaluer séparément les vrais K0 à K6 et les régressions contre S18 ; valider sur des compositions **non consultées depuis le début**. H9 reste interdit en tant qu'entrée. H8 reste inactif sans vrai audio pitch-shift et attestation OOF.

**Statut** : orientation architecturale explicite et testable, **pas** un run ni une solution confirmée. Les nouveaux résultats S38–S41 restent des diagnostics de développement, et la référence S18 ainsi que toutes les têtes déjà branchées sont conservées.