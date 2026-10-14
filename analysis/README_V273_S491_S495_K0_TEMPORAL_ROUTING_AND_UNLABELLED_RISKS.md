# S491–S495 — Boucles temporelles de récupération K0, preuves et limites de fiabilité

**Date : 14 octobre 2026. RECHERCHE SEULEMENT. AUCUNE PROMOTION, aucun changement à S80, freeze_local_combo, Live, onset/offset ou aux notes historiques.**

## Résumé causal

La substitution systématique S80→Basic Pitch apporte **12 corrections mais 7 régressions** sur les 45 vraies attaques guitare IDMT (S490). Le problème n'est donc plus seulement la précision du réseau, mais le **routage de la preuve d'une nouvelle note vers le bon événement natif V88**.

[S491 : run GitHub SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38576329909). Nouveau routeur *shadow* sans aucun accès aux annotations : ses seules entrées sont les **horodatages natifs des 506 événements guitare S80** (ou 575 événements piano V88), les **K d'origine**, et les **notes prédites / onsets** du modèle externe indépendant pré-entraîné. Une association un-à-un, distance ≤80 ms, est réalisée entre tous les événements V88 et les groupes d'attaques Basic Pitch. **Seuls les K0 originaux** peuvent changer. Deux politiques fixées a priori : `birth` (nombre de notes qui naissent dans le groupe, à 20ms) et `active30` (nombre de notes encore actives à l'instant natif +30ms). Tout K1..K6 d'origine est verrouillé.

[S492 : run GitHub SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38576853439). En post-gel :
- IDMT, 45 vraies attaques : **0 correction / 0 régression** sur les 2 politiques, 25/45 et 7/21 poly inchangés, tandis que **8 événements natifs sans annotation associée** sont passés de K0 à K positif. Ils ne sont **ni des vrais K0, ni des faux événements démontrés**.
- MAESTRO, 322 vraies attaques : `birth` fait **93 corrections**, 0 régression, **6 accords poly récupérés** ; 42→135/322 corrects ; **29** événements natifs hors vérités appariées passent également de K0 à K positif. Avec `active30` : **10 corrections**, 0 régression sur les attaques appariées, 42→52/322.
- Cette absence de régression **sur de vraies attaques ayant K≥1** est essentiellement une propriété de construction : le routeur ne peut modifier que les K0, qui étaient forcément erronés dans les 45/322 attaques annotées. **Aucune preuve zéro régression sur le flux complet**, faute d'annotations négatives vérifiées.

[S493 : run GitHub SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38577502262). Examen des **7 véritables attaques IDMT** que S80 avait classées K0 :
- les 7 ont une attaque Basic Pitch de la note véritable **à moins de 80ms de l'événement source**, malgré l'absence de proposition S491 ;
- les **7 sont en concurrence avec d'autres candidats V88** dans la même fenêtre de 80ms ;
- **6/7 événements V88 commencent plus de 30ms avant l'attaque XML**, dont 3/7 plus de 60ms ;
- pour 6/7 la note BP correcte est encore active à **XML+30ms**, mais seulement 1/7 à **l'horodatage V88 natif+30ms** ;
- **5/7** coïncident avec des notes soutenues dont le début estimé précédait déjà le début de la fenêtre natif de plus de 80ms. Ces faits démontrent un **défaut concret d'association temporelle**, pas une absence générale d'information de hauteur.

[S494 : run GitHub SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38578164091). Contre-proposition sans aucune vérité externe : associer chaque naissance Basic Pitch en priorité à un événement natif **K0 uniquement**, toujours un-à-un. Comparer `birth` et `activeAtBPonsetPlus30`, ce dernier décompte au vrai instant de l'attaque BP et non au temps V88 fréquemment avancé. Les K≥1 restent inchangés. Il en résulte **43 activations K0** sur 506 événements guitare, **215** sur 575 piano en mode `birth`. **Sur 35/43 activations guitare**, une prédiction originale K≥1 voisine est même *plus proche de l'attaque BP* que l'événement K0 choisi : danger explicite de **doublon et fausse note**, malgré le gain potentiel.

## S495 — Rejeu indépendant après blocage de GitHub Actions

Les tentatives de CI de S495 ont été interrompues par **HTTP 503 : “Egress is over the account limit”** lors du téléchargement d'un artefact Actions, *et non* par une erreur du modèle, après correction d'une première faute de syntaxe du script. Le verdict S495 n'est donc **pas un run GitHub SUCCESS**.

Pour conserver une preuve indépendante, j'ai utilisé le connecteur GitHub pour télécharger directement les **trois artefacts officiels** (S494, S488, S476), puis j'ai rejoué le calcul par événements dans un processus Python local. **Les empreintes SHA256 des fichiers sources à l'intérieur des archives concordent** avec les SHA enregistrés dans les rapports officiels. Les ensembles évalués restent exactement les **45 vraies attaques guitare** et les **322 vraies attaques piano**.

| S495, politique K0 PRIORITY `birth` | IDMT guitare, vraie chaîne S80 | MAESTRO piano, seulement V88 |
|---|---:|---:|
| Exact-K historique sur les attaques annotées | 25/45 | 42/322 |
| Exact-K proposé sur les mêmes attaques | **29/45** | **140/322** |
| Corrections sur attaques annotées | **4** | **98** |
| Régressions sur attaques annotées | **0** | **0** |
| K0 modifiés parmi attaques annotées | **5** (1 encore faux) | **141** (43 encore faux) |
| K0 activés parmi événements natifs sans vérité associée | **38** | **74** |
| Vrais accords polyphoniques Exact-K avant→après | 7→8/21 | 7→14/73 |

| S495, politique `activeAtBPonsetPlus30` | IDMT guitare | MAESTRO piano |
|---|---:|---:|
| Exact-K avant→après | **25→30/45** | **42→53/322** |
| Corrections / régressions sur attaques annotées | **5 / 0** | **11 / 0** |
| Vrais accords poly Exact-K avant→après | **7→9/21** | **7→14/73** |
| K0 activés parmi événements natifs NON annotés | **38** | **74** |

**ATTENTION :** aucun « 0 régression » sur les 45/322 attaques positives ne prouve une précision globale parfaite. C'est pour l'essentiel une propriété structurelle des corrections `K0→K positif` sur des vrais onsets connus comme porteurs d'au moins une note. Les **38 + 74 activations sans vérité associée** peuvent contenir des fausses notes, des résonances ou des événements non annotés ; elles ne sont pas admissibles comme corrections prouvées. Aucun seuil n'a été ajusté sur IDMT/MAESTRO pendant S491/S494, mais les deux jeux de test avaient déjà été exposés dans les audits précédents : **ceci n'est pas une nouvelle validation aveugle**.

## Décision

1. **Garder le routeur S494 comme prototype expérimental et rejeter sa promotion**. Le gain estimé 5 vrais onsets K0 retrouvés sur 7 est réel sur cette cohorte, mais 38 décisions K0 supplémentaires sont sans vérité négative établie et 35/43 ont une preuve possiblement déjà utilisée par un K positif voisin.
2. Le problème central à résoudre est maintenant **l'attribution et la déduplication des groupes d'attaques**, avec un seul objet musical événementiel par attaque réelle ; ensuite affiner K et les noms des notes, sans produire une seconde note à partir d'un sustain déjà représenté.
3. Construire une cohorte prospective réellement annotée **positifs ET absence de nouvel onset**, par guitare (différents microphones/réverbérations) et piano, et y figer les prédictions AVANT annotations. Mesurer *event precision* et *event recall*, Exact-K sur les attaques et note sets complètes ; ces trois métriques ne doivent pas être mélangées.
4. Respecter en permanence : **S80 exact-K officiel 81,98 % poly GuitarSet / 33,33 % sur 21 événements poly IDMT** et V88 MAESTRO ne sont pas les mêmes modèles, ni la même définition du K entre les jeux de données. S491–S495 ne modifient **aucun** de ces résultats officiels.

**Prochaine brique souhaitée : un agent de déduplication temporelle *multi-source onset graph*, entraîné sur des annotations d'attaques indépendantes avec supervision événementielle et des véritables négatifs, et veto des duplications avec source K≥1.**
