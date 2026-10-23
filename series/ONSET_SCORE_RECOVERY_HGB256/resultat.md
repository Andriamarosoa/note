# Résultat confirmé — récupération Onset HGB PCM256

Le replay des **80 WAV originaux** est terminé. Il couvre **306536 blocs de
256 échantillons** et reproduit exactement les **12281 attaques** prédites
avant le replay, pour chaque paire `(sample, pitch)`.

| Variante | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Contrôle69 historique | 8127 | 4519 | 3140 | 67,971396 % |
| Run 39915038301 | 8012 | 4264 | 3255 | 68,062694 % |
| **Nouvelle récupération HGB** | **8050** | **4231** | **3217** | **68,370987 %** |

Gain : **+0,308293 point de F1** contre le run précédent, **+38 TP / −33 FP /
−38 FN**. Contre le contrôle69 historique, le gain atteint **+0,399591 point**.
La précision passe de 65,265559 % à **65,548408 %** et le rappel de 71,110322 %
à **71,447590 %**.

## Résultats par composition

| OUTER | F1 précédent | Nouveau F1 | Delta points | Delta TP | Delta FP |
|---|---:|---:|---:|---:|---:|
| BN1 | 68,004587 % | 68,920455 % | +0,915867 | +27 | +5 |
| Funk2 | 71,903199 % | 72,160264 % | +0,257065 | −6 | −28 |
| Jazz1 | 71,207615 % | 71,772300 % | +0,564686 | +26 | +20 |
| Rock3 | 68,426225 % | 68,344519 % | −0,081706 | +7 | +22 |
| SS1 | 61,312314 % | 61,521833 % | +0,209520 | −16 | −52 |

50 pistes progressent, 24 régressent et 6 restent identiques en F1.
Le gain global ne masque pas la régression de Rock3 ni les pertes de TP sur
Funk2 et SS1. Aucun seuil n'a été retouché à partir de ces résultats OUTER.

## Modèles retenus

Les 120 modèles internes et 5 modèles finaux ont été entraînés. Tous les plis
ont choisi la politique `cautious` sur les compositions internes.

| OUTER | Modèle retenu | Seuil |
|---|---|---:|
| BN1 | HGB 15 feuilles | 0,45 |
| Funk2 | HGB 15 feuilles | 0,375 |
| Jazz1 | HGB 31 feuilles | 0,40 |
| Rock3 | HGB 31 feuilles | 0,40 |
| SS1 | HGB 15 feuilles | 0,375 |

Les modèles et leurs métadonnées sont conservés dans
`model/onset256_recovery_20261023/`. Le principal changement est l'apprentissage
sur les propositions réellement admissibles au bloc courant et leur étiquetage
par rapport aux attaques retenues par la tête principale figée. La capacité,
la politique et le seuil maximisent le F1 des quatre compositions internes.

## Vérification

- Sélection proxy de l'ancien module : parité exacte avec les sorties du replay
  précédent pour les 80 pistes, avant entraînement du nouvel expert.
- Nouveaux modèles sauvegardés avant les prédictions OUTER.
- Nouveau replay à partir du PCM original, poids et annotations vérifiés.
- Parité exacte des événements et des comptes TP/FP/FN avec les prédictions figées.
- 256 nouveaux échantillons par décision ; aucun audio futur, aucune attente
  ajoutée, aucune rétrodatation.
- **81 tests réussis** dans le
  [run GitHub 39923540639](https://github.com/Andriamarosoa/note/actions/runs/39923540639).
  Les nouveaux tests couvrent l'appariement un-à-un, la réattaque et la frontière
  numérique des seuils float32.

Le calcul audio complet est une exécution locale distincte du run GitHub de
tests. Rapports : `evidence/fresh_audio_report.json`,
`evidence/verified_summary.json` et `evidence/verification_environment.json`.
Code archivé au commit `6aebb36baa893a2f0c6f2c6e6e21c638d6d2a01c`.

## Hypothèse écartée et portée

La calibration multiple de la seule tête principale avait donné 67,680769 %
sur les événements archivés. Elle a été écartée avant un nouveau replay audio ;
son rapport est conservé dans `evidence/rejected_robust_main_report.json`.

La séparation de la nouvelle tête ne supprime pas les recouvrements historiques
des enseignants et des modèles amont. Le corpus FIT80 a déjà été réutilisé.
Ce résultat ne constitue pas une validation sur des enregistrements externes.
Les autres caractéristiques et le MIDI officiel ne sont pas promus par cette expérience.
