# V27.3 — correction native des positions et audit du fold 3

**Suite vérifiée :** le [correctif de couverture à 31 trames](v273-spectral-coverage-results.md)
résout les 96 attaques hors fenêtre décrites ci-dessous, sur les mêmes candidats
enregistrés. Ce rapport conserve les résultats historiques de l’étape précédente.

Verdict du 25 septembre 2026 : **les défauts de préparation identifiés sont
corrigés et la cohérence des nouveaux caches avec les données produites
directement à partir de l’audio est vérifiée. Aucun gain d’Exact K n’est encore
démontré.** L’audit complémentaire documente deux limites restantes : des
différences lors du recalcul des candidats et une couverture spectrale partielle
de certaines attaques annotées.

Périmètre : les 50 morceaux du seul fold externe 3, soit 15 279 groupes dont
1 969 polyphoniques. Aucun autre fold n’a été exécuté. Aucun entraînement,
réglage de seuil ou correcteur de sortie n’a été ajouté.

## Ce qui est corrigé et vérifié

| Contrôle | Résultat |
|---|---:|
| Origines des fenêtres audio corrigées | 82 groupes |
| Cibles de présence par corde corrigées | 16 groupes |
| Désaccords restants entre occupation des cordes et K | 0 |
| Différences entre caches corrigés et calcul direct sur les mêmes candidats bruts | 0 |
| Groupes tronqués dont les positions complètes sont maintenant conservées | 280 |
| Positions retirées des entrées du modèle mais conservées pour l’attribution des annotations | 2 009 |
| Cibles K différentes de l’ancien cache | 0 |

Les 82 corrections de fenêtres correspondent aux 80 erreurs de reconstruction
d’origine précédemment diagnostiquées, plus deux groupes dont le premier candidat
avait été retiré par la limite de 48 entrées. Pour ces deux groupes, l’utilisation
du premier candidat **retenu** décalait encore la fenêtre de 35 ou 120 échantillons.
Les 82 fenêtres et les 16 cibles modifiées sont toutes alignées avec des lignes
de l’ancien cache dont la géométrie et les cibles ont été vérifiées.

Les nouveaux caches V91/V100, en schéma 2, enregistrent les positions entières
d’origine, les positions retenues dans l’ordre des entrées du modèle et toutes
les positions avant troncature. Les cibles de cordes, de hauteur et de temps
utilisent les groupes complets. La fenêtre spectrale utilise l’origine originale.
Les anciennes données restent lisibles ; mélanger des caches anciens et nouveaux
est refusé. Produire un nouveau cache sans positions exactes est également refusé.

La validation réelle compare les deux formats après sauvegarde et relecture :
positions complètes, alignement des candidats retenus, cibles de cordes, hauteurs,
temps et cartes spectrales après leur conversion prévue en float16. Une attribution
indépendante des annotations retrouve les 8 812 attaques affectées aux groupes ;
454 des 9 266 annotations restent sans candidat dans le rayon d’attribution.

Deux autres erreurs de code sont couvertes par les tests : la comparaison erronée
d’un temps relatif à une origine absolue lors des collisions sur une corde, et
l’utilisation d’indices locaux au morceau dans la fonction V101 par défaut.
La correction de ce dernier défaut existait déjà dans un point d’entrée spécialisé.
**Aucune collision sur une même corde n’est observée dans ce fold** : le test
synthétique ne constitue donc pas une cause des erreurs de ce fold.

## Problèmes révélés par le contrôle complémentaire

### Le recalcul ne reproduit pas exactement les anciens candidats

Les caractéristiques de séquence diffèrent sur 4 821 lignes. Deux groupes ont
une différence structurelle, relevée explicitement après l’arrêt du premier
contrôle d’équivalence :

| Ancien indice, morceau | Différence observée | K ancien / nouveau |
|---|---|---:|
| 52 748 — `02_Rock3-117-Bb_solo.jams` | 30 candidats deviennent 29 ; disparition de la position relative 279 | 1 / 1 |
| 64 458 — `03_Jazz2-187-F#_comp.jams` | une position passe de 684 489 à 684 488, soit un échantillon | 0 / 0 |

Le nombre de groupes et toutes les cibles K restent identiques. **L’équivalence
complète des entrées avec l’ancien cache reste fausse.** Le mécanisme exact des
différences numériques et leur effet sur les prédictions ne sont pas établis.
Il serait incorrect d’interpréter un futur écart de score entre ces deux jeux
d’entrées comme l’effet isolé des corrections de temps. Une comparaison contrôlée
devra utiliser les mêmes propositions et caractéristiques figées pour ses deux
branches, en ne faisant varier que le traitement étudié.

### Certaines attaques affectées au groupe restent après la fenêtre spectrale

L’audit indépendant retrouve **96 attaques dans 87 groupes**, dont 42 groupes
polyphoniques, après la fin de la fenêtre ; aucune n’est avant son début.
La dernière se situe à 2 483 échantillons de l’origine du groupe, alors que la
fenêtre se termine à 1 764 échantillons, soit 40 ms.

La cause de cette couverture incomplète est précise : l’attribution accepte une
annotation située jusqu’à 882 échantillons, soit 20 ms, d’un candidat. Un candidat
proche de la fin d’un groupe peut donc recevoir une annotation au-delà de la fin
de la fenêtre spectrale. Réparer l’origine ne change pas cette règle.

Cette limite est auditée, **pas corrigée dans cette modification**. Les autres
entrées du modèle peuvent apporter de l’information sur ces attaques : leur
absence de la carte spectrale ne prouve pas à elle seule la cause d’une erreur K.
Leur effet sur le score reste à mesurer. Ces 42 groupes polyphoniques ne suffisent
pas à expliquer les 1 133 erreurs polyphoniques du contrôle précédemment audité.

## État de l’entraînement et preuves

Le correctif de préparation est disponible et vérifié. Un nouvel entraînement
n’a pas été lancé. La référence officielle V27.3 à 42,6019 % reste inchangée ;
elle ne doit pas être confondue avec le résultat d’un seul fold du contrôle reconstruit.
La couverture spectrale et la comparabilité des entrées doivent faire partie du
protocole du prochain essai, avec sélection interne et évaluation externe sur le
seul fold demandé.

- [Audit audio complet réussi — 36122007201](https://github.com/Andriamarosoa/note/actions/runs/36122007201), code `bdef8a92196e6f403748e1ed310750cdc7bed7d9` ; 17 tests réussis.
- [Audit complémentaire terminé — 36123987738](https://github.com/Andriamarosoa/note/actions/runs/36123987738), code `170e571f76d33b93d9b6ed11ed202b6a3d6962bd`. L’arrêt préalable du contrôle strict est conservé dans le run [36123656144](https://github.com/Andriamarosoa/note/actions/runs/36123656144).
- [Caches exacts et preuves conservées](https://github.com/Andriamarosoa/note/releases/tag/v273-exact-timing-36122007201).
- Rapports complets : [audit de préparation](v273-exact-timing-audit.json), [écarts et couverture](v273-exact-timing-population-audit.json).

SHA-256 de l’archive des caches : `5d40afdbf00bca3f3508907d1edf20856928b7cfcd0f89e58465fe3d359b8868`.
Les archives de rapports téléchargées ont également été vérifiées par leurs
empreintes avant publication. Les anciens poids et artefacts n’ont pas été remplacés.
