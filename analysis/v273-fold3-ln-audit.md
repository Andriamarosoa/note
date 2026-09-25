# Audit du surcomptage — fold 3 uniquement

**Suite exécutée :** la [correction et la décision livrées](v273-frozen-ln-results.md)
testent l'apprentissage de la branche seule avec contrôle figé. Les six
candidats sont rejetés en validation interne ; le paquet final et son
chargeur rétablissent les poids et seuils du contrôle, avec suppression
des 35 nouveaux surcomptages de cette piste sur le fold 3.

**Le mécanisme dominant sur ce fold est le changement des poids appris et
leur interaction avec les seuils. Dans 32 des 35 nouveaux surcomptages,
l'entrée ln est exactement nulle. Une fausse activation de cette entrée
au moment de ces prédictions ne peut donc pas expliquer ces 32 cas.**

Audit exécuté le 16 septembre 2026, sans entraînement, sur les modèles
sauvegardés du fold externe 3 : **15 279 exemples, dont 1 969 polyphoniques**.
Aucun autre fold externe n'a été évalué dans cet audit.

- [Exécution réussie](https://github.com/Andriamarosoa/note/actions/runs/35156618879).
- [Modèles sources](https://github.com/Andriamarosoa/note/releases/tag/v273-native-decay-35099819151).
- [Résultats, probabilités et caractéristiques conservés](https://github.com/Andriamarosoa/note/releases/tag/v273-fold3-ln-audit-35156618879).
- [Résultat machine](v273-fold3-ln-audit.json) et [empreintes des sources](v273-fold3-ln-audit-sources.json).

## Intervention vérifiée

Les six réseaux finaux sauvegardés ont été reconstruits et leurs poids
rechargés sous TensorFlow 2.15.1. Leurs prédictions ont été recalculées sur
les entrées acoustiques d'origine, dont les empreintes ont été vérifiées.
L'écart numérique maximal avec les probabilités archivées est de
**0,0000011921** ; **toutes les décisions finales des deux bras sont identiques**.

Dans les trois réseaux entraînés avec ln, l'entrée de 128 caractéristiques
a ensuite été remplacée par des zéros. Les autres entrées, l'ancre V104,
les poids et les seuils de ce bras sont restés fixes. Les empreintes des
poids avant et après inférence confirment leur absence de modification.
Les huit combinaisons d'activation des trois entrées ont été calculées.
Aucun seuil n'a été recherché sur les labels du fold 3.

## Résultat du retrait de ln

| Variante | Exact K polyphonique | Surcomptages polyphoniques | Sous-comptages polyphoniques |
|---|---:|---:|---:|
| Contrôle entraîné sans ln | 836 / 1 969 — **42,4581 %** | 275 | 858 |
| Modèle entraîné avec ln | 810 / 1 969 — **41,1376 %** | 298 | 861 |
| Même modèle avec son entrée ln coupée | 810 / 1 969 — **41,1376 %** | 297 | 862 |

Couper ln change 14 décisions polyphoniques : 5 erreurs sont corrigées,
5 bonnes réponses deviennent fausses et 4 erreurs changent de compte
sans devenir correctes. Le score exact reste donc identique.
Ce modèle amputé après entraînement est un diagnostic, pas un contrôle
réentraîné ni un modèle proposé au déploiement.

## Les 35 nouveaux surcomptages

Il s'agit des exemples polyphoniques corrects avec le contrôle et
surcomptés avec le bras ln : 18 vrais K=2 et 17 vrais K=3.
Ce nombre décrit une population d'erreurs nouvelles ; il diffère de la
hausse nette de 23 surcomptages, qui inclut aussi les autres changements.

- **32 / 35 ont un vecteur ln entièrement nul.** Le retrait de l'entrée
  laisse leurs probabilités et leur surcomptage inchangés.
- **3 / 35 ont une entrée ln non nulle et sont corrigés par son retrait** :
  indices globaux `37801`, `55786`, `66643`. Couper uniquement l'entrée ln
  de la tête V272 suffit à corriger ces trois cas.
- **12 / 35 sont corrigés en remettant seulement les seuils de transitions
  du contrôle**, avec les probabilités et le compte préalable du bras ln
  inchangés. Ces 12 cas sont distincts des 3 précédents. Cette vérification
  mesure l'effet des seuils existants ; elle n'en optimise aucun.

Le signal ln est non nul sur seulement 300 des 1 969 exemples polyphoniques
du fold. Un signal nul à l'inférence n'annule pas les changements de poids
acquis pendant l'entraînement. Les réseaux entraînés avec et sans cette
entrée peuvent ainsi produire des probabilités différentes sur un exemple
dont les 128 caractéristiques ln valent toutes zéro.

## Les deux erreurs prises précédemment comme exemples

Dans les **deux** cas ci-dessous, le vecteur ln est exactement nul et aucune
bande ne passe le filtre de fiabilité. Le compte avant les transitions
finales est correct dans les deux bras.

| Ligne globale | Vrai K | Contrôle | Modèle ln | Même modèle, ln coupé |
|---|---:|---:|---:|---:|
| 50105 — `00_Rock3-117-Bb_solo.jams` | 2 | 2 | 3 | 3 |
| 42907 — `02_Rock3-117-Bb_comp.jams` | 3 | 3 | 4 | 4 |

**Ligne 50105.** Le contrôle propose déjà 3, mais sa marge P(3)−P(2)
vaut 0,03153, sous son seuil de 0,16326. Le réseau entraîné avec ln produit
une marge de 0,15442 ; son seuil est abaissé à 0,10694, donc la règle
fait passer le compte de 2 à 3. Remettre le seuil du contrôle suffit ici
à conserver 2. L'entrée ln est nulle pendant toute cette comparaison :
la modification des probabilités vient des poids appris.

**Ligne 42907.** Le contrôle propose 3 ; le réseau entraîné avec ln propose
4 avec une marge P(4)−P(3) de 0,09170. Elle dépasse son seuil de 0,06234,
ainsi que le seuil du contrôle de 0,06107. Le changement des probabilités
apprises suffit ici à provoquer la révision erronée ; couper ln ou
remettre le seuil témoin ne la supprime pas.

## Correction de l'interprétation acoustique précédente

Le test sur des sons synthétiques a établi que l'indice peut être positif
sur des sons tenus sans nouvelle attaque. Ce constat de non-spécificité
reste valable. **Il ne démontre pas la cause dominante des erreurs réelles
de ce fold**, et ne peut pas expliquer une activation directe de ln pour
les deux exemples ci-dessus, puisque leur entrée ln vaut zéro.

L'audit isole un effet direct de ln sur trois nouveaux surcomptages et
un rôle des probabilités apprises et des seuils pour les autres.
Il ne détermine pas quel motif acoustique, ni quels exemples d'entraînement,
ont conduit aux changements de poids défavorables. Cela demanderait une
analyse de l'apprentissage, distincte du retrait à poids fixes effectué ici.

## Limites et reproduction

Le fold 3 a été choisi après observation de ses régressions. Les variantes
croisées sont des diagnostics après coup, sans validation indépendante.
Les limites du protocole initial concernant les sources candidates communes,
les données reconstruites et la seule configuration de graines persistent.
Le score historique V27.3 de 42,6019 % concerne une autre population et ne
sert pas de contrôle numérique à cette intervention.

```sh
PYTHONPATH=.:src python -B scripts/audit_v273_fold3_ablation.py --fold 3 \
  --cache-dir /chemin/caches_verifies \
  --fold-dir /chemin/fold-3 \
  --output-dir /chemin/audit_fold3
```

Le programme refuse tout autre numéro de fold. La restauration des sources,
les versions, les vérifications et l'archivage figurent dans
`.github/workflows/v273-fold3-ln-audit.yml` au commit
`08dfd5ee5c11b9320269607d68d236a59510013f`.
