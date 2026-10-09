# V27.3 Série29 — ordonnanceur dynamique après CHAQUE passage (A ou B→A)

**Protocole déposé avant le run.** Architecture retenue par l'utilisateur : l'ordre ne doit pas être le même pour tous les événements ; après chaque décision, l'ordonnanceur peut choisir une nouvelle tête, réutiliser l'état ou arrêter. La présente série réalise une première version opérationnelle de cette idée sur les **deux têtes A et B réellement implémentées**. Ce n'est pas encore le futur système de toutes les têtes spécialisées.

### État et actions

Pour chaque fragment sonore (59 309 événements, 19 morceaux, folds 0/1/2/4) l'état est `(signal X, parent S18, P_A(K0..6), message_B[7], variation P_A, étape)`. Le signal et l'état courant sont les **seules entrées** de la politique. Trois décisions successives au maximum. Au début et **après chaque action choisie**, la politique choisit :

- `STOP` : ne réévaluer aucune tête, conserver la proposition actuelle si sa marge probabiliste sur S18 dépasse 0,60 ; sinon conserver S18.
- `A` : recalculer A en conservant le message précédent de B (zéro au départ). B n'est **pas exécutée**.
- `B→A` : recalculer B à partir de l'état courant, puis recalculer A avec son nouveau message. B peut être exécutée **avant A dès le premier choix** ; aux suivantes, B reçoit les probabilités corrigées par A.

Une tête peut être revisitée, et chaque chemin peut être différent pour chaque événement, par exemple `A→BA→A`, `BA→A`, `A` ou `STOP`. Le même réseau A/B (19 checkpoints S20, 335 variables d'origine) est réutilisé à tous les passages. `BA` désigne deux appels *effectifs* (B puis A), et `A` un seul.

### Apprentissage et lutte contre les régressions

La politique est une estimation de l'utilité **conditionnelle à l'état après chaque passage**, et non un ordre préprogrammé. On explore systématiquement toutes les séquences de 0, 1 et 2 décisions parmi A/BA **sur les morceaux de formation uniquement** (1+2+4 états). Pour chaque état, les résultats possibles d'un nouvel A et B→A servent à former deux cibles séparées : `corrige S18` et `casse S18`, avec la vérité **uniquement sur morceaux de fit**.

On entraîne pour chaque morceau évalué, sur les autres morceaux de son même fold, six estimateurs Ridge multi-sortie (2 actions × 3 profondeurs), `alpha=200`, normalisation sur ces états de formation seulement, sans identifiant de morceau ou fold, sans vérité ni sortie YourMT3+ en entrée. Les données de formation sont préparées depuis les **propres checkpoints hors morceau** de leurs sources, avec limite de provenance explicitée plus bas.

Règle de politique prédéfinie `U(A)=P(fix_A)−lambda·P(break_A)−c_A`, `U(BA)=P(fix_BA)−lambda·P(break_BA)−c_BA`; `STOP` a U=0. Grille fixée lambda ∈ {1,2,4}, seuil d'engagement ∈ {0, 0,005, 0,02}. Coût exprimé en utilité : c_A=0,0005, c_BA=0,0010. Après chaque action, le score est **recalculé sur l'état réellement modifié**, et l'arrêt est irréversible. Ne jamais regarder la vraie classe de l'événement lors des choix.

### Audits et contrôles de conception

1. Test unitaire : A et BA produisent des distributions valides différentes, BA influence le prochain A, un STOP ne lance aucune tête, et les politiques de deux exemples avec états différents peuvent emprunter des chemins différents.
2. Pour toutes les 9 politiques : corrections, régressions et neutralités appariées à S18 et freeze ; global et poly K2–K6, vrais K0–K6, folds ; routes distinctes à chaque étape, chemins détaillés, nombres d'appels **réellement exécutés** à A et B et décisions finales par événement.
3. Contrôles fixes : S18 (0 appel), S20 complet A-first (4 A + 3 B), S25 B-first, S27 arrêt conditionnel, et S28 ordonnanceur choisissant une route seulement au départ. Un meilleur score global **sans** zéro régression vs S18 n'est pas une sélection promue.
4. Chaque modèle sauvegarde ses identités d'entraînement et de test et la provenance des 19 checkpoints ; archive intégrale, aucune promotion automatique.

**Avertissement important** : S18 a été choisi en explorant ces mêmes 59 309 événements, et les checkpoints S20 pour les autres morceaux ont parfois été entraînés avec des étiquettes du morceau actuellement évalué (contamination de *second niveau* possible pour l'apprentissage de l'ordonnanceur). L'exclusion du morceau de la régression ne suffit donc pas à établir une vraie validation sur nouvelles compositions. Les chiffres issus de cette série sont **exploratoires**, et devront être confirmés sur un ensemble de morceaux gardé hors de toutes les étapes de création des producteurs.
