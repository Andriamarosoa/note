# Audit énergétique V27.3 — source, transport, persistance et extinction

## Statut (8 octobre 2026)

Deux expériences CI terminées, sans changement de `freeze_local_combo` :

- [Audit K2/K3 de 488 erreurs et contrôles](https://github.com/Andriamarosoa/note/actions/runs/37737728309)
- [Audit élargi K2/K3/K4 de 1 800 groupes](https://github.com/Andriamarosoa/note/actions/runs/37738068677)

**IMPORTANT :** ni le fold 3 ni le joueur 05 n'ont été utilisés. Les folds
0/1/2/4 sont historiquement exposés et ne constituent pas un holdout neuf.

## Physique réellement implémentée

STFT de la sortie mono de la guitare, 24 bandes logarithmiques, contexte
[-80 ms, +160 ms], trames 512 échantillons et saut 128 échantillons.

Un facteur de survie passif par bande `a_b` est estimé uniquement sur le
pré-contexte (sans vérité terrain). La différence
`R_b=E_b(t+1)-a_b E_b(t)` est répartie, par convention, entre une
compensation locale de signes opposés sur bandes voisines (`J`),
une source résiduelle positive (`B`) et un puits résiduel négatif (`D`).

`R_b=J_{b-1/2}-J_{b+1/2}+B_b-D_b` est une **identité
construite** et numériquement vérifiée ; **pas** une conservation
d'énergie mécanique mesurée, ni une solution de Navier–Stokes.
`J` n'est pas identifiable de manière unique sur un canal mono.

## Audit K2/K3 sur les 488 cas de défauts connus

| Famille | AUC OOF | Net de corrections |
| --- | ---: | ---: |
| Énergie statique | 0.4908 | +1 |
| Naissance seule | 0.5132 | -11 |
| Transport seul | 0.5356 | -2 |
| Persistance seule | 0.5677 | +4 |
| Extinction seule | 0.5448 | +8 |
| Naissance + persistance + extinction | **0.6033** | +9 |
| Toutes dynamiques | 0.6009 | +13 |
| Temps brouillé | 0.4900 | -4 |
| Bandes permutées | 0.5658 | +5 |

La destruction de l'ordre temporel supprime l'avantage sur cette population
sélectionnée de cas difficiles. Le fold 2 reste négatif (net -5 avec
naissance+persistance+extinction). Le score de correction est exploratoire
et n'est pas une augmentation du pourcentage Exact-K global.

## Audit élargi K2/K3/K4

Échantillonnage prédéfini de 150 cas de chacune des classes par fold :
**1 800 cas, 600 par classe, 124 enregistrements**, supervision d'attaque
propriétaire du groupe natif. Classifieur logistique fixe à **trois classes**
avec trois folds en entraînement et un fold tenu hors entraînement,
puis permutation des folds. **Ne pas comparer** directement avec le
modèle complet à sept classes.

| Famille | Précision équilibrée | Macro AUC OVR | Rappel K2 | K3 | K4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Statique | 46.50 % | 0.6457 | 59.5 % | 23.2 % | 56.8 % |
| Naissance + persistance + extinction | 46.56 % | 0.6545 | 51.7 % | 32.5 % | 55.5 % |
| Dynamiques seules | 47.06 % | 0.6572 | 52.2 % | 33.2 % | 55.8 % |
| Statique + dynamiques | 47.50 % | 0.6840 | 55.5 % | 31.5 % | 55.5 % |
| Temps brouillé | 46.22 % | 0.6600 | 58.5 % | 28.8 % | 51.3 % |
| **Bandes permutées** | **48.78 %** | **0.6861** | 58.5 % | 32.2 % | 55.7 % |

Les perturbations ne constituent pas un test causal parfait :
le brouillage temporel modifie les frontières entre phases ; la
permutation des bandes préserve certaines statistiques et modifie
la géométrie supposée.

### Décision

1. L'information temporelle est observable, mais son bénéfice pour K2/K3/K4
   reste faible, hétérogène et non validé indépendamment.
2. L'hypothèse **d'un transfert local entre bandes log-fréquentielles**
   n'est **pas soutenue** par ce test, car la permutation des bandes n'est
   pas pénalisée. Elle n'équivaut pas à un flux d'énergie acoustique spatial.
3. Notre proxy n'améliore pas le rappel K4 face au simple état énergétique.
4. Aucune modification du réseau de référence, aucune promotion, aucune
   utilisation de player 05 ou fold 3.
5. Prochaine direction : étudier un couplage des **trajectoires harmoniques
   et des sources dans le temps**, plutôt que des bandes voisines arbitraires ;
   évaluer indépendamment naissance/tenue/étouffement sur audio synthétique
   avec vérité terrain par source, puis sur compositions tenues hors recherche.

## Code et artefacts

- `scripts/extract_v273_energy_transport.py` ;
  `scripts/summarize_v273_energy_transport.py`
- `scripts/extract_v273_energy_transport_multik.py` ;
  `scripts/summarize_v273_energy_transport_multik.py`
- `test/test_v273_energy_transport.py`
- `.github/workflows/v273-energy-transport.yml`
- `.github/workflows/v273-energy-transport-k234.yml`
- Artefacts : `v273-energy-transport-summary` et
  `v273-energy-transport-k234-report` sur les deux runs GitHub liés
  ci-dessus. Les deux runs ont terminé avec succès.
