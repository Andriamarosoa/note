# Défaut de conception : une cible globale, des entrées locales incomplètes

**Le contrat du comptage natif utilisé par nos derniers tests est incomplet :
K dépend des candidats des groupes voisins, mais leur géométrie complète
n’est pas transmise au modèle.** Un contre-exemple exécuté avec les fonctions
du dépôt produit les quatre mêmes tenseurs d’entrée pour deux cibles
différentes. Modifier la normalisation ou le nombre d’époques ne fournit
pas cette information manquante.

Ce défaut d’observabilité est démontré. **Le gain d’Exact K qu’apportera sa
réparation reste à mesurer.** Il ne constitue pas une explication de toutes
les erreurs.

## Le décalage dans le code

La cible compte les attaques **attribuées au groupe**, pas toutes les notes
audibles dans la fenêtre. Chaque attaque annotée est affectée au groupe du
candidat le plus proche, à une distance maximale de **882 échantillons
(20 ms)**. En cas d’égalité, le plus petit identifiant de groupe gagne.

| Élément | Dépendance vérifiée |
|---|---|
| [`_assign_refs`](../scripts/evaluate_v90_cluster_oracles.py) | Compare tous les groupes du morceau pour attribuer chaque attaque. |
| [`_cluster_arrays`](../scripts/train_v90_structured_cluster_cardinality.py) | Transmet au plus 48 candidats et huit statistiques du groupe courant. |
| [`batch_inputs`](../scripts/v273_window_experiment.py) | Fournit `candidate_set`, `candidate_mask`, `cluster_stats`, `spectral_map`. |
| [Mémoire V8.7](../scripts/train_v87_causal_candidate_memory.py) | Contexte de candidats passés ; aucun candidat du groupe suivant. |

Les positions complètes existent dans les caches mais ne sont pas toutes
des entrées du réseau. Le spectre peut montrer une attaque attribuée au
voisin sans révéler quelle proposition concurrente gagne cette règle.

## Preuve : mêmes entrées, deux bonnes réponses

Le test utilise les fonctions existantes de regroupement, d’attribution,
de construction des caractéristiques et de préparation des entrées. Les
représentations acoustiques et figées des candidats locaux restent constantes.
Seule la position d’une proposition du groupe suivant change.

Temps relatifs en échantillons à 44 100 Hz : candidats courants **0 et 1 764**,
attaque annotée à **2 600**. Sa distance au candidat courant est donc 836.

| Cas | Candidat du groupe suivant | Distance à l’attaque | Groupe gagnant | K courant |
|---|---:|---:|---|---:|
| A | 3 300 | 700 | Suivant | **0** |
| B | 3 500 | 900 | Courant | **1** |

Les quatre tenseurs natifs et les séquences de mémoire des deux candidats
locaux sont **strictement identiques**. Un modèle déterministe recevant ces
seules entrées doit répondre la même chose aux deux cas ; il ne peut pas
satisfaire les deux cibles.

**Portée :** contre-exemple au niveau de l’interface des propositions et
caractéristiques figées. Ce n’est ni une paire de doublons trouvée dans
GuitarSet, ni une exécution des réseaux amont depuis deux audios complets.
Il établit la dépendance omise ; il ne chiffre pas un plafond d’accuracy réel.

## Le délai fait partie de la correction

La fenêtre acoustique actuelle finit à **2 788 échantillons**, soit 63,22 ms
après le début du groupe. Les concurrents du contre-exemple sont plus tardifs.

Si `e` est le dernier candidat courant, une borne conservatrice de complétude
des propositions pour attribuer toutes les attaques admissibles est
**`e + 2 × 882`** : une attaque peut être à `e + 882`, et un concurrent plus
proche encore jusqu’à près de 882 échantillons après elle.

Pour un groupe occupant les 40 ms autorisées, cela donne **3 528 échantillons,
soit 80 ms**, avant tout délai supplémentaire de production des propositions.
Si l’on exige aussi leurs caractéristiques V8.6 complètes, leur horizon de
1 024 échantillons porte la borne pour ces caractéristiques à **103,22 ms**.
La disponibilité réelle doit être vérifiée dans le producteur de propositions.

La fréquence des cibles réelles qui changeraient avec un délai limité n’est
pas mesurée. Un candidat gagnant tardif ne prouve pas à lui seul ce changement :
un autre candidat déjà disponible du même groupe peut encore gagner.

## Présence dans les preuves réelles du seul fold 3

L’archive acoustique précédente a été relue et vérifiée par empreintes.
Sur les **874 surcomptages K<4** du modèle natif uniforme à 31 trames :

- **262** comportent une attaque admissible localement mais attribuée ailleurs.
- **139** impliquent le groupe suivant, 126 le précédent, avec trois cas communs.
- Dans **140** cas, la prédiction égale le nombre d’attaques admissibles localement,
  avant concurrence entre groupes.
- **612** surcomptages ne présentent pas cette concurrence locale.

Exemple conservé : `01_BN1-147-Gb_comp.jams`, indice global **17020**.
Le modèle prédit 2 pour une cible de 1. L’attaque à l’échantillon 460059
appartient au groupe courant 162. Celle à 460539 est à seulement 299
échantillons d’un candidat courant, mais appartient au groupe suivant 163.
Deux attaques sont localement admissibles ; une seule doit être comptée.

Ces observations montrent que la situation existe. **262 n’est ni un nombre
d’erreurs causalement expliquées ni une promesse de 262 corrections.**

## Corrections proposées

### 1. Fournir l’appartenance au groupe avant le comptage — priorité

Conserver les cibles, fournir les positions complètes des candidats locaux
et concurrents et calculer les zones d’appartenance avec la règle exacte.
Cette entrée utilise les propositions et leurs horodatages, jamais les
annotations. Garder l’audio voisin comme contexte ; le supprimer brutalement
retirerait aussi de l’information utile.

Le composant [`v273_ownership_context.py`](../scripts/v273_ownership_context.py)
est écrit et testé. Il calcule l’appartenance aux instants demandés, distingue
A de B et refuse un contexte non certifié jusqu’au délai requis. Il doit
recevoir les groupes complets, avant la limite de 48 lignes. Une complétude
de propositions certifiée n’est pas simplement un nombre d’échantillons lus.

**Intégration restante :** transmettre cette géométrie au réseau comme
contexte temporel. Une compression en fractions par trame doit être auditée
aux frontières ; elle n’est pas automatiquement équivalente aux positions
exactes. Fournir hors ligne des voisins encore indisponibles en inférence
créerait une fuite temporelle. Si le délai requis est incompatible avec
l’application, une autre règle causale de groupes/cibles doit être définie
et évaluée comme un nouveau protocole, sans comparaison directe à l’ancien score.

### 2. Relier K à des attaques localisées et attribuées

Le chemin [`train_v250_count_only.py`](../scripts/train_v250_count_only.py)
retire explicitement les sorties et pertes d’événements, de classement et
de localisation. Il apprend seulement K. Cette propriété est vérifiée ;
son effet causal sur les scores n’est pas mesuré par la preuve précédente.

Après correction du contexte, superviser les **nouvelles attaques et leur
appartenance**, puis imposer la cohérence entre le compte et les événements
distincts. On pourra distinguer attaque manquée, double détection et mauvaise
attribution, au lieu de ne disposer que d’une mauvaise classe K.

Ne pas sommer naïvement les 48 propositions : elles peuvent être redondantes,
comme l’a montré l’[audit V17.7](v177-post-audit.md). Des têtes d’événements
ont déjà été essayées dans les versions précédentes. La correction spécifique
ici est le **contexte d’appartenance explicite et disponible au bon moment**,
pas seulement le retour d’une ancienne tête.

## Validation et état réel

Premier test : ajouter seulement le contexte d’appartenance, avec les mêmes
cibles, normalisation historique, objectif, données, initialisation commune
et budget. Témoin et traitement doivent partager le même délai de décision.
La supervision d’événements fera l’objet d’un test séparé.

Rester dans la partition interne associée au fold 3. Rapporter les cas
avec/sans concurrence, K=0/1/2/3 séparément, corrections et régressions,
score polyphonique et latence. La correction du contrat doit réussir les
contrôles géométriques avant de mesurer le gain d’Exact K.

**Sept tests passent** : contre-exemple, rayon et égalités, équivalence avec
l’attribution existante sur 300 requêtes, complétude temporelle, accord d’un
préfixe certifié avec tous les candidats, effet d’une troncature et rejet
des entrées invalides. Le helper n’est pas encore connecté à un modèle
entraîné. Aucun nouveau score, entraînement ou remplacement de V27.3 n’est
revendiqué par cet audit.

[Diagnostic et empreintes](v273-design-contract-audit.json). Reproduction :

```sh
PYTHONPATH=.:src python -B -m unittest -v test.test_v273_ownership_context
PYTHONPATH=.:src python -B scripts/audit_v273_design_contract.py \
  --evidence /path/to/acoustic-audit --output /path/to/design-contract-audit.json
```

Archive : [audit acoustique du fold 3](https://github.com/Andriamarosoa/note/releases/tag/v273-acoustic-residual-36373588730).
