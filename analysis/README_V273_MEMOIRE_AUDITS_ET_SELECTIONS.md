# V27.3 — Conserver chaque apport et rendre les régressions exploitables

Le contrat est désormais explicite : **un apport particulier suffit à conserver
une variante comme sélection candidate, même si son bilan global est inférieur**.
Son parent, ses corrections, ses régressions et ses cas neutres restent accessibles.
Un gain observé constitue une preuve de cet apport sur les événements concernés ;
il ne prouve pas encore que le réseau saura reconnaître ces événements sur de
nouvelles données.

Cette étape corrige une perte d'information dans les entrées du réseau et crée
une mémoire vérifiée des variantes disponibles. Elle ne prétend pas que toute
l'histoire du dépôt est déjà transformée en têtes exécutables, ni que toutes les
régressions sont maintenant bloquées.

## 1. Ce qui a effectivement changé

Les cinq canaux d'audit local étaient calculés et archivés, puis mis à zéro
avant l'entrée du catalogue global : taux locaux de correction/régression/neutre,
support local et distance des voisins. Deux variantes séparées, à sept et huit
sélections, les reçoivent maintenant effectivement. Les anciennes restent présentes.

Les votes, les propositions des **127 ou 255 groupes complets**, et les neuf
descripteurs d'audit bruts sont identiques à ceux des modèles initiaux. Les
producteurs, la perte, les paramètres, la seed 27402 et les 30 époques restent
ceux du protocole annoncé avant résultats. Aucune mauvaise performance d'un
membre seul ne supprime ses combinaisons.

[Protocole préalable](README_V273_PROTOCOLE_MEMOIRE_AUDITS.md) ;
[exécution terminée](https://github.com/Andriamarosoa/note/actions/runs/37807124287),
code exécuté `a287fbf42f091fa996b1f085416d64b28a7b381e`.

## 2. Les apports particuliers survivent au bilan global

Résultats face à `freeze_local_combo`, sur les mêmes 59 309 événements natifs :

| Variante | Corrections | Régressions | Gain net |
|---|---:|---:|---:|
| Catalogue 7, audits globaux | 582 | 522 | +60 |
| Catalogue 7, audits locaux activés | 536 | 504 | +32 |
| Catalogue 8, audits globaux | 584 | 546 | +38 |
| Catalogue 8, calibration précédente | 516 | 459 | +57 |
| Catalogue 8, audits locaux activés | 527 | 495 | +32 |

Le dernier bilan ne justifie pas de supprimer le catalogue local. Face au
catalogue 8 global, il conserve 470 corrections, en perd 114 et **en apporte
57 nouvelles**. Il conserve 443 régressions, **en évite 103** et en introduit
52. Ainsi, 160 décisions deviennent justes et 166 deviennent fausses, soit −6.
Ces 160 améliorations sont conservées comme preuves distinctes du gain net.

Le catalogue 7 local conserve 496 corrections et en apporte 40 nouvelles ; il
évite 75 anciennes régressions. Ses 115 améliorations et 143 dégradations sont
également conservées malgré le bilan −28 face à son parent.

La version calibrée du catalogue 8 corrige 80 erreurs initiales que la version
globale ne corrige pas ; celle-ci en corrige 148 que la calibration laisse passer.
La calibration améliore au total 251 décisions du parent et en dégrade 232.

Les trois versions du catalogue 8, globale/calibrée/locale, corrigent au moins
une fois **707 erreurs initiales différentes**. Chacune possède encore des cas
uniques parmi les trois : **65 / 66 / 43**, respectivement. Cette union est un
diagnostic de complémentarité calculé avec les labels, **pas le résultat d'un
sélecteur qui saurait déjà choisir la bonne version**.

## 3. Les audits locaux influencent réellement le réseau

Après l'entraînement local, une intervention remet seulement les cinq canaux
locaux à zéro, sans réentraîner les poids :

| Réseau local | Gain avec canaux à zéro | Gain avec canaux présents | Améliorations / dégradations dues à leur rétablissement |
|---|---:|---:|---:|
| Catalogue 7 | +4 | +32 | 224 / 196, soit +28 |
| Catalogue 8 | −24 | +32 | 248 / 192, soit +56 |

Cela prouve une influence des entrées locales sur ces modèles. Le +56 n'est
**pas** une amélioration face au catalogue global réentraîné séparément : cette
comparaison vaut −6. Les sorties de l'intervention sont elles aussi archivées
comme candidates mesurées ; même celle au bilan −24 n'est pas effacée.

La surestimation persiste : le catalogue 8 local annonce un gain cumulé de
354,45, pour +32 observé, contre 397,42 pour +38 auparavant. La perte
probabiliste sur les événements évalués passe de 1,07749 à 1,08599. L'activation
des audits ne constitue donc pas à elle seule une solution à la surestimation.

## 4. Une régression persistante rouvre sa catégorie

Pour chaque verdict effectivement choisi, l'audit examine **tous les groupes
complets proposant ce verdict**, et compare leurs taux locaux de correction
et de régression. Le premier masque de groupe affiché par le décodeur n'est pas
une attribution causale : les groupes de même K final partagent le score final.

Voici tous les changements du catalogue 8 local, et pas seulement ses erreurs :

| Profil local des groupes proposant le verdict | Corrections | Régressions | Neutres |
|---|---:|---:|---:|
| Au moins un défavorable, aucun favorable | 92 | 126 | 123 |
| Groupes favorables et défavorables mélangés | 347 | 280 | 445 |
| Au moins un favorable, aucun défavorable | 88 | 88 | 129 |
| Tous à égalité | 0 | 1 | 1 |
| **Total** | **527** | **495** | **698** |

Un blocage uniforme de la première ligne éviterait 126 régressions, mais
supprimerait aussi 92 corrections. Ce constat est un diagnostic a posteriori,
sans ajout d'une règle en production. Il montre que ce profil agrégé ne suffit
pas à distinguer les succès des échecs. Les 126 régressions malgré une preuve
défavorable montrent aussi que fournir cette preuve comme entrée n'impose pas
au réseau de respecter le risque.

Il faut donc conserver le fragment, sa transition K initial → K proposé,
les groupes, les caractéristiques observables et les preuves M/N/neutre ; puis
départager, sur des références autorisées, une catégorie trop large, un signal
insuffisamment utilisé, un support inadéquat ou une caractéristique absente.
Une sous-catégorie doit être reconnaissable sans lire le label du cas évalué.
**Une nouvelle erreur est une obligation d'audit, pas une preuve automatique
d'appartenance à une nouvelle catégorie.**

Les combinaisons restent nécessaires : sur 17 erreurs initiales du catalogue 8,
un groupe peut donner le vrai K alors qu'aucune des huit sélections seules ne
le propose. Le modèle global en corrige 5, le local 4. Ces cas et les réussites
propres à chaque variante sont conservés ; un veto individuel les détruirait.

## 5. Mémoire durable et vérifiée

[Registre des variantes](evidence/v273-audit-memory/variants/registry.json) :

- **17 archives sources explicitement recensées**, avec tous leurs champs de
  décision diagnostiques natifs, donnent **41 sorties candidates**, dont
  **30 vecteurs de prédiction distincts**. Les doublons sont identifiés et gardés.
- Chaque sortie conserve ses 59 309 verdicts, ses résultats par rapport au
  freeze et au parent, ainsi que ses corrections/régressions/neutres par vrai K.
  **Trois candidates au gain global négatif sont conservées.**
- Les 7 493 événements éligibles sont reliés à leur enregistrement, leur
  position et aux **58 caractéristiques primaires d'origine** :
  **434 594 valeurs float64 vérifiées identiques**. Les labels d'audit sont
  stockés séparément de ces caractéristiques observables.
- Les **1 573 erreurs initiales corrigées par au moins une candidate** sont
  listées avec toutes les candidates justes et fausses. C'est une union
  descriptive des réussites, pas une performance de généralisation.
- Les corrections et les régressions de chaque variante restent reconstituables
  dans `all-variant-decisions.npz`. Les CSV des changements locaux conservent
  également les profils de succès, d'échec et de neutralité.
- Les poids et rapports disponibles des 17 sources, les paramètres de
  calibration et les deux caches de producteurs sont archivés avec leurs
  empreintes SHA256. Le contenu est vérifié après décompression. Ces fichiers
  sont conservés dans le dépôt, au-delà de la rétention des artefacts CI.

L'[inventaire historique](evidence/v273-audit-memory/source-archives/history-summary.json)
recense **27 923 entrées**, dont **180 patches de correction**, sur **54
références Git lues sans erreur**. Les scripts `learn_`, les modules auxiliaires,
les preuves CSV/NPZ, les tests et les workflows entrent maintenant dans son
périmètre. Les identifiants et les SHA des fichiers de branche sont conservés.
Cet inventaire décrit les têtes de branches récupérées et les commits de
correction retenus par le collecteur ; il ne prétend pas couvrir toutes les
versions de tous les fichiers de toute l'histoire. Ses entrées ne sont pas
27 923 têtes exécutables ni autant d'idées indépendantes.

Les archives de paramètres ne dupliquent pas les gros tenseurs de probabilités
de toutes les exécutions passées : leurs sources restent référencées par SHA256.
La mémoire durable conserve tous leurs verdicts natifs recensés et les 58
caractéristiques primaires ; elle ne remplace pas une sauvegarde exhaustive
de tous les signaux audio et de leurs transformations.

## 6. Ce qui est établi, et ce qui reste à intégrer

**Établi :** les apports particuliers ne sont plus filtrés par le bilan global ;
les cinq canaux locaux sont effectivement connectés dans deux variantes ;
leurs effets, leurs échecs persistants et leurs complémentarités sont mesurés
et conservés. Les tests de conservation passent, avec 13 tests dans les jobs
modèles et 6 tests locaux ciblés, dont les 2 tests de mémoire ajoutés ensuite.
Les prédictions, la provenance des producteurs, les exclusions et tous les
résultats publiés sont recalculés par les vérificateurs.

**Encore nécessaire pour l'intégration :** les 41 sorties archivées ne sont
pas encore 41 nouvelles entrées actives du réseau. Il faut un adaptateur et
des prédictions imbriquées pour chaque candidate, avec exclusion de tous les
labels interdits au modèle consommateur. Les versions calibrées utilisent des
labels d'autres morceaux du fold exclu ; elles ne peuvent pas être directement
injectées dans un protocole qui interdit ce fold entier. Les quatre folds
0/1/2/4 ont déjà été examinés. Fold3/player05 restent exclus ; aucune validation
sur données inédites ni promotion n'est revendiquée.

La protection contextuelle attendue reste à démontrer : reconnaître sur des
cas indépendants quand une sélection ou une combinaison doit être écartée,
tout en préservant ses cas utiles. Aucun résultat de cette étape ne permet
d'annoncer que toutes les régressions connues ou futures sont bloquées.

## Rejouer les vérifications

Les chemins `--artifacts-root` et `--features` désignent les archives originales
énumérées dans le registre. Les deux outils de création refusent d'écraser une
mémoire existante.

```bash
python -m unittest test.test_v273_variant_memory test.test_v273_audit_context test.test_v273_history_preservation
python -m scripts.build_v273_variant_memory --artifacts-root ../ --features ../selector-review-artifacts/features --output /tmp/v273-memory-new
python -m scripts.verify_v273_variant_memory --memory analysis/evidence/v273-audit-memory/variants --artifacts-root ../ --features ../selector-review-artifacts/features --output /tmp/v273-memory-verification.json
python -m scripts.verify_v273_audit_memory --results ../audit-memory-results --original ../catalogue-results --features ../selector-review-artifacts/features --output /tmp/v273-audit-verification.json
python -m scripts.archive_v273_memory_sources --artifacts-root ../ --output /tmp/v273-source-archive-new
```

Preuves : [audits et exclusions](evidence/v273-audit-memory/verification.json),
[conservation exacte des variantes](evidence/v273-audit-memory/variant-memory-verification.json),
[manifestes des paramètres et de l'inventaire](evidence/v273-audit-memory/source-archives/manifest.json),
[artefacts de l'exécution](evidence/v273-audit-memory/artifacts.json).
