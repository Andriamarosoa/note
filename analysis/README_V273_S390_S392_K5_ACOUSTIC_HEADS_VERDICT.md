# S390–S392 : audit des nouvelles têtes acoustiques K5

13 octobre 2026. Référence de **travail** : S80 reconstruit S272/S273. Aucun checkpoint S80 n'a été modifié. Les résultats de S80 sur GuitarSet ont une contamination par l'entraînement : ne pas assimiler les bilans descriptifs aux gains de généralisation.

## Evidence

- [S389 — plafond de couverture K5](https://github.com/Andriamarosoa/note/actions/runs/38350310670). Parmi 355 vrais K5, S80 S272 en détecte 99 ; 224 sont prédits K4. YourMT3+ détecte 189 K5. 124/137 cas reconnus seulement par YourMT3+ sont K4 selon S80.
- [S390 — deux nouveaux producteurs audio 104D/1133D](https://github.com/Andriamarosoa/note/actions/runs/38350679942). Huit checkpoints, apprentissage dans les autres folds, aucune intersection de morceaux en train/test. En entraînant sur les lignes où S80 historique OOF prédit K4 : au seuil fixe 0,50, audio104 = +46 K5/−115 K4, net −69 ; audio1133 = +16 K5/−21 K4, net −5 sur S80 S272 descriptif.
- [S391 — huit nouveaux producteurs 1133D à cible reformulée](https://github.com/Andriamarosoa/note/actions/runs/38351071250). Apprentissage à partir des vrais K4/K5 (pas des prédictions historiques), ou des vrais K3/K4/K5. À seuil 0,30, vue K4/K5 = +69 K5/−140 K4, net −71. À 0,20, vue K3/K4/K5 = +25 K5/−42 K4, net −17 sur S80 S272.
- [S392 — diagnostic reproductible de séparabilité](https://github.com/Andriamarosoa/note/actions/runs/38351303318). Parmi **1 115** événements où S80 prédit K4 : **224 vrais K5, 806 vrais K4, 85 autres**. Taux K5 = 20,09 %.

| Classement des vrais K5 parmi 1 115 S80→K4 | ROC AUC | Average precision |
|---|---:|---:|
| S390, audio104 | 0,6462 | 0,2698 |
| S390, audio1133 | 0,6613 | 0,2921 |
| S391, entraînement K4/K5 | **0,6730** | 0,2847 |
| S391, entraînement K3/K4/K5 | 0,6540 | 0,2906 |

Les probabilités K5 / K4 se chevauchent fortement : aucune de ces quatre têtes statiques ne fournit un seuil mesuré à gain net positif contre S80 S272 (le non-déclenchement a net 0). Le problème n'est donc **pas seulement** le choix de seuil ou l'ordre des correcteurs. Il s'agit, sur cette population, d'un défaut de séparabilité du cinquième événement à partir des 104D/1133D statiques et des modèles essayés. Cela **ne prouve pas** qu'aucun autre modèle statique n'y arriverait.

## Conséquence technique

Ne pas greffer les quatre têtes à S80. Préserver intactes leurs preuves, leur entraînement et les poids pour audits contradictoires.

Prochaine expérience de conception recommandée : **proposer explicitement une cinquième attaque/voix à partir d'un signal temporel**, pas simplement convertir après coup le cardinal K4 en K5. Comparer un producteur count-aware qui tient compte du contexte temporel et de l'enveloppe d'attaque, à un contrôle sans séquence, en fold-heldout, en imposant la conservation des vrais K4. Les candidats K5 devront être émis **en amont de S72** puis évalués sous les règles de S80.

Critère de promotion de tout correctif ultérieur : corrections K5 **moins** régressions des vrais K4/K0–K3 sur un ensemble de morceaux indépendants, et score global/polyphonique réellement amélioré. Aucun des lots GuitarSet/IDMT déjà ouverts ne redevient « inédit » après ajustement.

**Statut : diagnostic terminé, quatre architectures statiques sans gain net contre S80; S80 reste la référence de travail.**
