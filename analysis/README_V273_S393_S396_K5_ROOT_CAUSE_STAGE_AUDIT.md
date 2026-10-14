# S393–S396 : localisation vérifiée de la perte K5 dans S80

Date : **13 octobre 2026**. Audits réels sur **59 309 événements GuitarSet alignés**, dont **355 vrais K5**, avec les vrais K fournis uniquement au scoring final. La reconstruction S272/S273 est **avec recoupement entraînement**, pas un contrôle indépendant.

## 1. Deux niveaux d'explication à distinguer

**Perte logicielle de la classe K5 : localisée.** Sur les **224 vrais K5 prédits K4 par S80**, le parent **S18 avait initialement prédit K5 sur les 224**. Cette information est principalement perdue par la politique de sélection S70.

**Cause acoustique de la cinquième note : NON PROUVÉE.** Le benchmark n'inclut ici qu'une vérité de cardinalité K, pas l'identité MIDI/pitch de la cinquième note manquée. Les descripteurs de trajectoire 42x49 sont harmoniques, pas des pistes de notes indépendantes.

## 2. S393, banque PR16 originale (36 sorties) et trajectoire temporelle

[Workflow S393](https://github.com/Andriamarosoa/note/actions/runs/38351764311), terminé avec succès.

- Parmi les **224** vrais K5 perdus en K4 : **186 n'ont AUCUNE proposition K5 dans les 36 sorties PR16**, 38 en ont au moins une, 7 ont le choix PR16 policy22 à K5.
- En parallèle, parmi les **806 vrais K4 bien classés**, 123 ont au moins une proposition PR16 K5.
- La médiane du rapport énergie d'attaque/avant attaque est **2,632** dans les K5 perdus contre **4,181** dans les K4 corrects ; fraction de nouveauté à l'attaque **0,660** contre **0,786** ; nombre médian de pistes harmoniques émergentes à seuil 20 % est **5 vs 5**. Ce sont des corrélations descriptives, confondues par le morceau et les différences d'événements.

**Avertissement** : le chiffre antérieur de « seulement 109 propositions K5 » dans S389 concernait un sous-ensemble de *sorties sélectionnées*, et pas tout le catalogue S46 ni S18. Il ne fallait pas en faire un plafond du système entier.

## 3. S394, toutes les 193 sorties S46

[Workflow S394](https://github.com/Andriamarosoa/note/actions/runs/38352001951), terminé avec succès.

Pour les **224 K5→K4**, les **193 sorties S46** proposent toutes K5 en médiane ; leurs valeurs S18 sont un parent susceptible d'être *recopié* par fallback, pas 193 détections indépendantes. Sur **118/224** de ces cas, les **193 sorties sont identiques au parent**. L'« unanimité » n'établit donc pas 193 confirmations acoustiques. La provenance S267–S268 confirme la règle de retour au parent S18 et le recoupement avec les données d'entraînement.

## 4. S395, localisation EXACTE de la première perte

[Workflow S395](https://github.com/Andriamarosoa/note/actions/runs/38352208180), terminé avec succès.

Parmi les 224 vrais K5 finalement K4 :

| Étape réellement exécutée | Nombre encore prédits K5 |
|---|---:|
| S18 parent | **224** |
| S69 polymax | **202** |
| S70 proposition poly | **2** |
| S71 poly | **2** |
| S72 poly poly_guard | **0** |
| S79 puis S80 | **0** |

Première perte par événement : **22** à S18→S69, **202** à S69→S70. Dans cette population, S70 récupère aussi 2 K5 auparavant perdus dans S69, puis S72 les retourne à K4. Les **deux dernières actions S72** sont des pertes secondaires, pas 2 erreurs de plus au-delà des 224.

**Conclusion de conception vérifiée :** le problème majeur de cette population n'est PAS l'absence universelle d'un producteur K5 ; S18 avait reconnu K5, et **la tête méta-sélectrice S70 remplace massivement K5 par K4**. Dans le code, S70 reconstruit des variables à partir des **12 réseaux PR16, des 36 votes, de S18 et des parents S69**, puis une régression logistique par fold propose un K différent et l'applique lorsque **3 des 4 modèles sont d'accord**. La règle `proposed_poly` n'interdit pas K5→K4 (elle autorise toute cible K >= 2 différente du parent).

## 5. S396, contrefactuel de garde STRUCTUREL (diagnostic seulement)

[Workflow S396](https://github.com/Andriamarosoa/note/actions/runs/38352355580), terminé avec succès. On conserve S80 et ne fait qu'un *replay contrefactuel des prédictions finales*, sans réexécuter les modèles S71/S72.

| Règle hypothétique | Actions | K5 corrigés | Régressions | Net global |
|---|---:|---:|---:|---:|
| Protéger un K5 rétrogradé K4 par S70, si S80 est K4 | 203 | **202** | 0 | **+202** |
| Protéger un K5 rétrogradé K4 par S69 | 22 | **22** | 0 | **+22** |
| Union des deux | 225 | **224** | 0 | **+224** |

Une des 225 actions sur le premier garde est un vrai K1 initialement déjà incorrect ; elle reste incorrecte en K5 et ne compte donc ni comme correction ni comme régression.

**INTERDICTION de présenter +224 comme une amélioration indépendante :**
- la conception du garde suit **l'analyse des erreurs du même jeu GuitarSet** ;
- S18/S46 et S70, puis S80 reconstruit, utilisent des modèles avec **recoupement de données d'entraînement** ;
- ce type de résultat 224 bonnes réponses sur 225 modifications, très favorable et inhabituel, est justement un signal supplémentaire de contamination / manque de représentativité ;
- la simulation n'a **pas** rejoué S71/S72 après modification de S70 ;
- la composition-disjonction et la stabilité sur de nouveaux morceaux ne sont pas démontrées.

## 6. Décision d'ingénierie

**Conserver S80 comme référence de travail inchangée.** Corriger prioritairement la politique de **déclassement 5→4 dans S70**, et auditer la règle S69 de même nature. Éviter de produire de nouveaux K5 par simple post-correction tant que le chemin S18→S70 perd des propositions déjà disponibles. Les 16 modèles S390/S391 sont archivés comme contrôles négatifs.

Pour une modification réelle : faire un nouveau run complet S70→S71→S72→S80 avec une règle inscrite avant notation, sur **compositions inédites avec assez de K4/K5** et annotations complètes. Mesurer corrections, régressions global/poly, résultats par K et comparabilité de contexte avec YourMT3+. Pour expliquer la *cause acoustique* d'une cinquième note, il faut une vérité MIDI par hauteur et le WAV original de chaque cas.
