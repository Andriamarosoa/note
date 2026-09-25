# Audit des erreurs du contrôle V27.3 — fold externe 3

Demande : vérifier les causes des erreurs qui subsistent après le rejet de ln,
puis auditer chaque problème identifié. Le périmètre reste un seul fold externe.

## Référence et unité mesurée

Le contrôle vient de l'expérience native reconstruite `35099819151`, fold 3.
Le modèle exporté `v273-frozen-ln-35159220367` reproduit ses 15 279 décisions,
dont 836 comptes corrects parmi 1 969 groupes polyphoniques. Ce résultat de
développement ne reproduit pas le score historique de 42,6019 %.

**K compte les débuts de notes annotés attribués à un groupe de candidats,
avec plafonnement à 6. Il ne compte pas toutes les notes encore en résonance.**
Les candidats sont regroupés sur 40 ms et les débuts de notes attribués au
groupe le plus proche à au plus 20 ms d'un candidat. Cette définition corrige
l'explication précédente en termes de notes jouées simultanément.

## Vérifications des décisions

1. Vérifier les empreintes et l'alignement des archives du fold 3, des trois
   réseaux du contrôle, de l'ancre V104 et du paquet exporté.
2. Rejouer toutes les décisions avec un décodeur scalaire indépendant ; exiger
   l'égalité avec les fonctions de production et les prédictions enregistrées.
3. Comptabiliser les corrections et régressions de chaque étape et de chaque
   transition. Une erreur transformée en autre erreur n'est pas une correction.
4. Partitionner sans chevauchement les 1 133 erreurs polyphoniques : bonne
   proposition du spécialiste bloquée par le seuil 0/1, une transition absente,
   ou une marge insuffisante ; bonne proposition d'un réseau complet seulement ;
   aucun des trois réseaux de comptage ne propose la vérité.
5. Effectuer les seules ablations déclarées : retrait du sauvetage, retrait
   des transitions ensemble ou individuellement, remplacement polyphonique
   conditionné par K prédit >=2, spécialiste sur toutes les lignes pour exposer
   son incapacité à produire 0/1. Aucun de ces résultats ne sélectionne un modèle.

Les prédictions et seuils ont déjà été inspectés pour construire ce diagnostic.
Il ne s'agit ni d'une expérience aveugle ni d'une validation de généralisation.
Les comptes de propositions correctes utilisent les annotations : ils décrivent
une possibilité a posteriori, pas un routage utilisable sans vérité terrain.

## Audit des problèmes que ce diagnostic fait apparaître

- **Routage** : identifier chaque décision bloquée et compter les régressions
  lorsque l'on retire une protection. Préserver les métriques K=0/1.
- **Seuils** : comparer le bilan interne archivé au bilan externe, sans choisir
  de nouveaux seuils sur le fold 3 et sans assimiler une variation à une preuve
  de surapprentissage.
- **Cibles et entrées** : restaurer les huit archives natives vérifiées, mais
  analyser les entrées et prédictions du seul fold externe 3. Les histogrammes
  des labels d'apprentissage sont descriptifs, sans évaluation des autres folds.
- **Annotations** : utiliser l'archive GuitarSet déjà employée, vérifiée par
  l'empreinte MD5 historique et archivée dans le rapport avec SHA-256. Refaire
  l'attribution des débuts de notes aux candidats reconstruits avec deux
  implémentations, puis comparer compte, occupation des cordes et cache.
- **Fenêtre spectrale** : mesurer combien de débuts de notes attribués dépassent
  la fenêtre de -29,66 à +40 ms du spectrogramme. Les caractéristiques des
  candidats sont une autre entrée ; une note hors de cette seule fenêtre ne
  prouve donc pas que toute information sur cette note est absente du modèle.

Les candidats conservés sont reconstruits à partir de positions float16 ;
les labels exacts d'origine peuvent avoir été produits avant troncature des
candidats. Les divergences seront rapportées avec cette limite. Aucune écoute
ni séparation de sources n'est effectuée dans cet audit.

## Décision

Aucun entraînement, changement de poids, réglage de seuil, nouveau correcteur
ou promotion automatique. Chaque constat publié doit préciser sa preuve et
son statut : mécanisme vérifié, association observée, ou cause encore inconnue.
Les corrections du modèle nécessiteront une validation interne indépendante
du diagnostic externe ; les explications factuellement erronées peuvent être
corrigées immédiatement.

## Suivi ciblé après le premier audit

L'exécution `36105248716` a trouvé 16 divergences de comptes, réparties
en huit paires voisines. Le total des débuts de notes est conservé. Les
positions reconstruites présentent aussi des espacements incompatibles
avec la construction des groupes. Un cas synthétique reproduit un défaut
de `_recover_cluster_start` : plusieurs origines expliquent exactement les
mêmes meilleurs candidats, et le dernier critère choisit la plus proche de
zéro, sans preuve que ce soit l'origine réelle.

Le suivi ajoute une contrainte provenant des scores fusionnés sauvegardés :
le premier `top_sample` doit correspondre à un candidat dont le score float16
est maximal. Une origine diagnostique n'est changée que si cette contrainte
et les positions conservées n'admettent qu'une solution. Les annotations
ne participent jamais au choix de cette origine. Les ambiguïtés restantes
sont rapportées et conservées ; aucun spectrogramme ni modèle n'est remplacé.

Les contrôles portent sur la cohérence avec le cache V91 avant génération
spectrale, la réversibilité des 1 765 positions relatives entières à travers
float16, l'espacement des groupes non tronqués et l'accord avec les comptes
originaux après attribution indépendante des annotations. Le score du modèle
n'est pas recalculé sur des données modifiées et aucun gain n'est annoncé.
