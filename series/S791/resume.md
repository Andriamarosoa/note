---
serie: S791
statut: en_cours
type: experience
reference: S708
branche: codex/note-s791-framepitch-complementary-experts
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S791 — Résumé

## Objectif

Corriger la conception insuffisante de Frame Pitch S790 : le nombre de têtes spécialisées doit être **déterminé par les défauts acoustiques** et non fixé artificiellement à un ou six. Évaluer des experts réellement complémentaires et un choix dynamique par fragment, puis retenir uniquement ceux apportant un gain reproductible sur FIT.

## Hypothèse

Les erreurs de hauteur sont hétérogènes : graves E2–B2, voisins harmoniques, continuité Sustain/Release et ambiguïtés générales. Une banque de spécialistes de tâches différentes avec sélection causale au niveau du fragment devrait proposer davantage de corrections utiles que la seule tête globale S790.

## Modifications

- Banque expérimentale multi-têtes à domaines différents : référence S708, généraliste, graves, discrimination des partiels harmoniques et persistance des notes.
- Sélecteur dynamique par fragment utilisant uniquement les observations acoustiques déjà reçues ; sélection/composition des experts auditée sur FIT80.
- Mesure de la contribution et de la non-redondance de chaque expert : un candidat sans apport doit être écarté.
- Tests de causalité par perturbation des signaux futurs et invariants de la référence S708.
- Six audits séparés et consensus bloquant, `resume.md` et `resultat.md` maintenus.

## Protocole

Comparer les experts sans ouvrir TEST ni HELD. FIT80 avec séparation par cinq compositions pour entraîner et choisir ; CAL32 diagnostic figé une fois le sélecteur arrêté. Le score de la tête Frame Pitch à 1024 échantillons reste un **shadow**, pas le F1 officiel de la sortie MIDI à hop 256.

La banque ne fixe pas six experts comme règle : elle **évalue quatre spécialisations motivées** et doit garder uniquement les experts présentant un bénéfice OOF reproductible. Une tête entraînée sur les mêmes données, sans autre fonction, ne compte pas comme nouvelle expertise indépendante.

## Audits obligatoires

- Données : qualité PCM/annotations, répartition par composition, classe et registre.
- Acoustique : fondamentales graves, voisinage harmonique et état temporel.
- Architecture : spécialisation réelle, routeur par fragment, indépendance des paramètres.
- Apprentissage : OOF par composition et détection des experts sans contribution.
- Métriques : micro/macro F1, TP/FP/FN, erreurs par registre et comparaison S708.
- LIVE : aucune connaissance du futur ni altération des événements MIDI S708/S707.

## Statut

En cours. Nouveau score : non mesuré. S790 reste une expérience distincte ; S708 reste la référence officielle.
