# V27.3 — fenêtre spectrale corrigée et auditée sur le fold 3

Verdict du 25 septembre 2026 : **le défaut de couverture est corrigé dans le
nouveau format natif à 31 trames : 96 attaques attribuées mais hors fenêtre
deviennent zéro.** Les candidats, caractéristiques et cibles de comptage sont
strictement conservés. **Aucun gain d’Exact K n’est encore démontré.**

L’audit porte sur les 50 morceaux du seul fold externe 3 : 15 279 groupes,
dont 1 969 polyphoniques. Aucun autre fold n’est évalué, aucun modèle n’est
entraîné sur GuitarSet et aucun correcteur de sortie n’est ajouté.

## Cause et correction

L’attribution accepte une attaque à moins de 882 échantillons d’un candidat.
Un groupe peut s’étendre sur 1 764 échantillons. Une attaque attribuée peut donc
se situer jusqu’à **2 646 échantillons après l’origine**, alors que la carte
spectrale historique s’arrête à 1 764. La carte ne montrait pas toutes les
attaques que le modèle devait compter.

La nouvelle fenêtre se termine à 2 788 échantillons : le délai de regroupement
(1 764) plus le délai de vérification déjà déclaré (1 024). Elle conserve le
précontexte de 1 308 échantillons et passe de 23 à 31 trames. Cette durée vient
des bornes du protocole, pas d’une optimisation sur les annotations du fold 3.

La décision nécessite donc l’audio jusqu’à **63,22 ms après l’origine du groupe**.
Cette extension reste dans le délai acoustique maximal annoncé par V100/V102 ;
elle ne démontre pas une absence de délai et ne mesure pas la latence réelle
du système en fonctionnement.

## Résultats mesurés

| Contrôle | Ancien format exact, 23 trames | Nouveau format, 31 trames |
|---|---:|---:|
| Attaques attribuées hors fenêtre audio | 96 | **0** |
| Attaques attribuées hors intervalle des centres de trames | 273 | **0** |
| Annotations attribuées | 8 812 | 8 812 |
| Annotations sans attribution | 454 | 454 |
| Groupes | 15 279 | 15 279 |
| Groupes polyphoniques | 1 969 | 1 969 |

Les 96 attaques récupérées concernent **87 groupes, dont 42 polyphoniques**.
Pour chaque morceau, l’audit vérifie après sauvegarde et relecture :

- égalité octet par octet des 14 champs conservés : caractéristiques de
  candidats, masques, statistiques, identités, positions exactes, limites de
  groupes, troncature, cibles K et présence par corde ;
- égalité octet par octet des 23 premières trames spectrales ;
- égalité du calcul spectral du cache et de l’API runtime sur les mêmes
  candidats complets, après la conversion float16 prévue ;
- mêmes hauteurs, présences et positions physiques des attaques attribuées ;
- distributions temporelles normalisées sur les 31 trames. Leur support
  gaussien change volontairement ; il ne faut pas les décrire comme inchangées.

Les candidats viennent des fichiers exacts déjà conservés. **Aucun nouveau
recalcul de propositions** n’est intervenu. Les différences numériques observées
lors du précédent recalcul ne contaminent donc pas cette comparaison 23/31.

Les bords des fichiers sont contrôlés : 50 groupes ont un précontexte complété
par des zéros et 24 ont une fin de fenêtre complétée par des zéros. Aucune attaque
attribuée n’est hors de l’audio réel. La présence de ces zéros n’est pas présentée
comme une observation acoustique.

## Compatibilité et audit des problèmes introduits

Le nouveau schéma 3 décrit explicitement la fenêtre à 31 trames. Les anciens
formats restent à 23 ; mélanger les fenêtres ou des métadonnées incohérentes est
refusé. La génération de la fenêtre étendue exige des positions exactes.
Les coordonnées temporelles des 23 premières trames restent elles aussi inchangées.

Le premier essai a échoué avant l’audit audio : un import historique remplaçait
le constructeur V102 par sa variante qui exploite la masse attribuée aux sources,
laquelle n’acceptait pas encore le paramètre de durée. Cette incompatibilité
**introduite par l’extension** a été corrigée ; elle n’est pas invoquée comme
cause des erreurs de comptage antérieures. L’échec est conservé dans le run
[36139128485](https://github.com/Andriamarosoa/note/actions/runs/36139128485).

Le second essai vérifie les vrais constructeurs transitifs des modèles de
comptage à 7 et 5 classes : paramètres initiaux identiques à 23/31 trames,
probabilités normalisées, gradients finis et mise à jour synthétique effective.
Ces contrôles emploient uniquement du bruit synthétique et des classes factices,
**aucun exemple externe pour l’apprentissage**. Les pertes historiques de
localisation d’événements ne sont pas migrées : à 31 trames, le constructeur
V240 est explicitement limité à la construction des modèles de comptage.

24 tests ont réussi dans Actions, puis 25 localement après ajout du rejet
explicite des anciens caches ambigus pour la fenêtre étendue. Ce dernier garde-fou
ne change pas le traitement des caches exacts du run réussi.

## Portée du verdict et preuves

Ce défaut d’entrée est résolu ; **les erreurs de comptage globales ne le sont
pas encore**. Les 42 groupes polyphoniques concernés ne suffisent pas à expliquer
les 1 133 erreurs polyphoniques du contrôle audité précédemment. Les 454
annotations sans attribution sont conservées dans le bilan, sans changement
de règle destiné à améliorer artificiellement le résultat.

Un essai d’apprentissage contrôlé reste nécessaire pour mesurer l’effet sur
Exact K, avec sélection interne et fold 3 réservé à l’évaluation. Les nouveaux
fichiers de ce fold servent à l’audit ou à l’évaluation, jamais à l’ajustement.
La référence officielle **V27.3 à 42,6019 %** reste inchangée.

- [Run réussi 36139446276](https://github.com/Andriamarosoa/note/actions/runs/36139446276), code `438e19c736080bf1c735bb8eb15ef687e4bbfc86`.
- [Rapport complet des 50 morceaux](v273-spectral-coverage-audit.json).
- [Vérification des modèles](v273-spectral-coverage-model-checks.json).
- [Caches natifs et preuves conservés](https://github.com/Andriamarosoa/note/releases/tag/v273-spectral-coverage-36139446276).

SHA-256 de l’archive native : `e885a2e100296e2ce257690b704432097869c2c7883654214bd4f9859ede4742`.
SHA-256 du rapport audio : `b36871b8c30739caa18136c8e41c87d91dae4cd409b288c74fafe3c8b4fc6d21`.
L’archive Actions téléchargée est vérifiée contre son empreinte
`c6391b49731e8e06e9ae4ab84ccc7a2b42f6a57d2885d10d243a1dfab212963d`.
