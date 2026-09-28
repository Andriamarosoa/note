# Surcomptages K < 4 : audit de l'audio et de l'attribution des attaques

Audit élargi à la cause des faibles performances : [apprentissage, objectif
et distinction entre K](v273-performance-causes.md). Il compare maintenant
les modèles sur leurs exemples vus et non vus ; le présent document reste
l'analyse acoustique ciblée des surcomptages.

**Audit terminé, uniquement sur le fold 3. La pondération explique
l'aggravation mesurée précédemment. Les 874 surcomptages qui subsistent sans
pondération ont plusieurs formes : concurrence entre groupes, comptes trop
grands même sans annotation étrangère, et signal réel sans note annotée.**
L'audit identifie ces situations et écarte un écart entre audio et cache ; il
ne démontre pas une cause acoustique unique ni une correction validée.

Périmètre : composant natif de comptage, 31 trames, poids uniformes,
prédictions figées de l'expérience `36351028493`. Il s'agit du composant
utilisé dans V27.3, pas d'une nouvelle mesure de la chaîne V27.3 complète.
Les prédictions supérieures à 3 sont conservées lorsque le vrai K est inférieur
à 4. Aucun réentraînement, correcteur de sortie ou nouveau seuil.

## Résultat par vrai K

| Vrai K | Groupes | Surcomptages | Taux | Avec attaque admissible mais attribuée ailleurs | Sans aucune annotation étrangère dans la fenêtre |
|---|---:|---:|---:|---:|---:|
| 0 | 10 285 | 315 | 3,06 % | 116 | 48 |
| 1 | 3 025 | 292 | 9,65 % | 83 | 65 |
| 2 | 806 | 201 | 24,94 % | 57 | 58 |
| 3 | 628 | 66 | 10,51 % | 6 | 26 |
| Total | **14 744** | **874** | **5,93 %** | **262** | **197** |

Les deux dernières colonnes décrivent des sous-ensembles des erreurs ; elles
ne constituent pas une décomposition exhaustive de leurs causes.
Sur ces mêmes groupes, le modèle produit 12 653 comptes exacts et
1 217 sous-comptages. Faire simplement baisser K ne constitue donc pas une
solution démontrée.

## 1. Une concurrence concrète dans la définition de la cible

Une attaque annotée est attribuée à un seul groupe : celui dont un candidat
est le plus proche, à une distance maximale de 882 échantillons, soit 20 ms.
En cas d'égalité, le premier groupe gagne. Deux groupes peuvent donc être
assez proches pour voir et revendiquer la même attaque, alors qu'un seul reçoit
le crédit dans sa cible K.

**262 des 874 surcomptages, soit 29,98 %, contiennent au moins une attaque
admissible pour le groupe courant mais attribuée à un autre.** Dans
140 cas, le nombre d'attaques localement admissibles correspond exactement à
la prédiction trop haute. Cette concordance est compatible avec une difficulté
à respecter l'appartenance au groupe ; elle ne prouve pas que le réseau
compte explicitement toutes ces attaques.

La comparaison avec les groupes sans concurrence est indispensable :

| Vrai K | Surcomptage avec concurrence | Surcomptage sans concurrence |
|---|---:|---:|
| 0 | 116 / 961 = 12,07 % | 199 / 9 324 = 2,13 % |
| 1 | 83 / 252 = 32,94 % | 209 / 2 773 = 7,54 % |
| 2 | 57 / 181 = 31,49 % | 144 / 625 = 23,04 % |
| 3 | 6 / 94 = 6,38 % | 60 / 534 = 11,24 % |

L'association est nette à K=0 et K=1, mais elle n'est pas universelle : elle
s'inverse à K=3. Les groupes sont issus d'un seul fold et ne constituent pas
des observations indépendantes permettant une généralisation automatique.

La fenêtre contient une attaque d'un autre groupe dans 302 erreurs au total,
dont les 262 cas de concurrence ci-dessus. Ces nombres se chevauchent.
Supprimer les groupes concurrents dans le calcul des cibles changerait la
tâche : aucun score recalculé avec cette autre règle n'est annoncé.

## 2. Les voisins et les notes déjà actives ne suffisent pas

**149 surcomptages à K=1, 2 ou 3 ne contiennent aucune annotation étrangère
au groupe dans la fenêtre** : 65 à K=1, 58 à K=2 et 26 à K=3. Toutes les
notes annotées qui chevauchent ces fenêtres appartiennent au groupe courant.
Il reste donc une erreur de cardinalité même dans ce sous-ensemble défini
par les annotations. Cela ne garantit pas l'absence physique de tout autre son.

Sur l'ensemble des 874 erreurs :

- Dans 211 cas, la prédiction égale le nombre maximal de notes annotées
  simultanément actives. C'est une concordance descriptive, pas une preuve
  que les notes tenues sont la cause de chacune de ces erreurs.
- Dans 309 cas, la prédiction dépasse même ce maximum simultané.
- Dans 227 cas, elle dépasse le nombre total de notes annotées qui
  chevauchent la fenêtre, simultanées ou non.

Ces ensembles se chevauchent avec d'autres catégories et ne doivent pas être
additionnés comme des causes exclusives. Le nombre de notes actives ne
remplace jamais la cible, qui compte les nouvelles attaques attribuées.

## 3. K=0 sans annotation n'est pas du silence

Il existe 347 groupes K=0 sans aucune note annotée chevauchant la fenêtre.
**48 sont surcomptés par le modèle uniforme et aucun des 48 ne contient un
signal PCM entièrement nul.** Le niveau RMS médian de ces erreurs est
−31,03 dBFS. Une absence d'annotation ne justifie donc pas de qualifier ces
fenêtres de silencieuses.

Exemple vérifié : `03_Funk3-98-A_solo.jams`, début à **14,99340 s**, indice
global `51287`. La cible vaut 0, le modèle prédit 1 avec une probabilité de
97,97 %, aucune annotation ne chevauche la fenêtre, et le signal RMS vaut
−26,02 dBFS. La forme d'onde contient une montée d'amplitude et des oscillations.
Cette observation ne permet pas, à elle seule, de trancher entre événement
musical non annoté, autre transitoire ou limites temporelles des annotations.

Le frontal spectral code des contrastes relatifs au signal précédent,
notamment `ln(1 + puissance / référence_précédente)` et le flux positif.
Un contraste spectral important n'est pas une preuve d'une nouvelle note
appartenant au groupe. **L'audit n'établit pas que le logarithme, le bruit ou
les harmoniques sont la cause générale des surcomptages.**

## 4. Exemples supplémentaires vérifiables

Exemples choisis par confiance décroissante dans chaque K, non représentatifs
de la fréquence des erreurs :

| Piste et début | Cible → prédiction | Constat issu des annotations |
|---|---:|---|
| `00_Funk3-98-A_comp.jams`, 6,13846 s | 1 → 2 | Cinq attaques localement admissibles, dont quatre attribuées au groupe voisin. |
| `04_Funk3-98-A_comp.jams`, 33,05746 s | 2 → 4 | Deux notes annotées, toutes deux attribuées au groupe ; aucune annotation étrangère. |
| `03_Jazz2-187-F#_comp.jams`, 5,13315 s | 3 → 4 | Trois notes annotées, toutes trois attribuées au groupe ; aucune annotation étrangère. |

Les 874 traces conservent les probabilités, les débuts et fins d'annotations,
les cordes, le groupe propriétaire et la distance aux candidats. Les huit
extraits illustratifs sont disponibles dans l'archive publiée.

## Vérifications et portée du verdict

- Les archives audio et annotations sont vérifiées contre leurs empreintes
  originales. Les 50 pistes appartiennent exclusivement au fold 3.
- Les 9 266 annotations ont été réaffectées indépendamment : 8 812 attribuées,
  454 non attribuées, et les 15 279 cibles K sont reproduites exactement.
- Les spectres des **874 erreurs et 667 témoins corrects** ont été recalculés
  depuis l'audio original : **1 541 égalités exactes avec le cache float16**.
  Cette vérification ne prétend pas couvrir les spectres de tous les groupes.
- Les 15 279 lignes et décisions ont été comparées localement à l'audit
  précédent. Les statistiques du rapport, les 874 traces et les mesures
  RMS/crête des huit extraits ont été revérifiées indépendamment.
- Les dix tests ciblés du protocole ont réussi dans le workflow. Les
  suppressions de caractéristiques déjà étudiées restent des sondes hors
  distribution, pas des corrections validées.

**Cause démontrée de l'aggravation : la pondération de l'apprentissage.**
Dans l'expérience contrôlée précédente, elle fait passer les surcomptages
K<4 de 874 à 1 434 à 31 trames, soit +560. Voir
[l'audit précédent](v273-low-k-audit.md).

**Pour les erreurs résiduelles, l'audit documente une ambiguïté d'attribution
et une difficulté de cardinalité qui persiste sans annotation étrangère.**
Le réseau prédit directement K à partir du spectre et des candidats ; son
graphe réduit ne vérifie pas explicitement qu'une attaque comptée appartient
au groupe. Le rôle causal de cette limite architecturale nécessite une
expérience native dédiée, conservant les mêmes cibles. Pour les 48 erreurs
sans annotation, l'origine physique et l'exhaustivité des annotations restent
à établir ; elles ne sont pas déclarées résolues.

Le prochain test pertinent doit donc distinguer l'appartenance d'une attaque
au groupe et le nombre d'attaques distinctes, avec une comparaison contrôlée
sur partitions internes. Ce rapport ne lance pas cet entraînement et ne
valide aucune nouvelle architecture. **V27.3 reste la référence officielle à
42,6019 %**, non directement comparable aux mesures de ce composant isolé.

## Sources et reproduction

- [Workflow terminé avec succès](https://github.com/Andriamarosoa/note/actions/runs/36373588730).
- [Archive complète, traces et extraits](https://github.com/Andriamarosoa/note/releases/tag/v273-acoustic-residual-36373588730).
- [Rapport brut](v273-acoustic-residual.json), [diagnostic recalculé](v273-acoustic-residual-diagnosis.json), [provenance et empreintes](v273-acoustic-residual-sources.json).
- Source de l'auditeur : `ca360975f17ce10709eb6c3d2e0bbbec2ac0db7c`.

Après extraction vérifiée des archives acoustique et du précédent audit :

```sh
PYTHONPATH=.:src python scripts/summarize_v273_acoustic_residual.py \
  --root /path/to/acoustic-audit --previous /path/to/low-k-audit \
  --output /path/to/diagnosis.json
```
