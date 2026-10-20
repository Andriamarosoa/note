---
serie: S803
statut: en_cours
type: experience
reference: S708
branche: codex/note-s803-causal-current-frame-gates
run_id: 39361632654
commit_sha: 1cd2c3d8c6e4795f1c4ce4425cf7211c426e26ac
promotion: false
---

# S803 — résultats GuitarSet FIT80 réels, rejets et audits

## Exécution

[Run scientifique #39361632654](https://github.com/Andriamarosoa/note/actions/runs/39361632654) au commit `1cd2c3d8c6e4795f1c4ce4425cf7211c426e26ac`. Le replay MIDI256 a effectivement traité **80/80** pistes GuitarSet FIT80 OOF, avec SHA256 des snapshots S800, vérification de 12 compteurs S795, des artefacts S801/S802 et reproduction complète de la référence FIT. L'artefact réel s'appelle `S803_REAL_GUITARSET_FIT80_MIDI256_METRICS_AUC`.

L'exécution initiale [#39361223391](https://github.com/Andriamarosoa/note/actions/runs/39361223391) a réussi le replay, mais les six audits ont échoué **avant l'audit** par `ModuleNotFoundError: torch` : une dépendance PyTorch inutile au script d'audit léger. Ce défaut CI a été corrigé dans la branche sans modifier le replay, les modèles ni le décodeur. Le deuxième run a réussi le replay et les six audits indépendants. L'appel NVIDIA du deuxième run a subi une panne/expiration sans réponse exploitable ; les jobs en échec ont été relancés dans ce même run (tentative 2). Ne pas inscrire 7/7 PASS sans preuve du consensus final dans les artefacts du même run.

## Métriques

Scores en % : **comparaison appariée à la source S795/S800 FIT80**. Ce tableau n'est pas une évaluation CAL32 et ne remplace pas S708.

| Métrique | Référence S708 CAL32 | Source FIT80 | rise_attack FIT80 | parent_conflict FIT80 | agreement_attack FIT80 |
|---|---:|---:|---:|---:|---:|
| Onset F1 | 74,2639 | 73,2520 | 72,9168 | 72,8779 | 73,2009 |
| Exact-K | 62,0455 | 61,6972 | 61,7034 | 61,7125 | 61,6883 |
| Active Exact-K | 56,2762 | 57,0195 | 57,0304 | 57,0416 | 57,0087 |
| Offset accuracy | 77,3292 | 69,8862 | 69,4622 | 69,3865 | 69,8424 |
| Full Note F1 | 57,1319 | 50,6528 | 50,1154 | 50,0342 | 50,5858 |
| Frame Pitch F1 | 76,4626 | 78,7278 | 78,8056 | 78,8147 | 78,7418 |

Évaluations à six mesures : **zéro candidat accepté**. `rise_attack` et `parent_conflict` bloquent 0 cellule additionnelle par rapport à leurs modes S801 respectifs et reproduisent les mêmes régressions. `agreement_attack` bloque **99** cellules (19 NoteOn synthétiques restants), mais régresse sur **5/6** mesures et sur **14** paires composition×métrique ; aucun compromis n'est admissible au veto strict. `rise_attack` a 118 NoteOn synthétiques et 19 paires composition×métrique en baisse ; `parent_conflict` en a 136 et 17 paires en baisse.

Diagnostic acoustique rétrospectif, vrais étiquetages d'attaques S802, aucune utilisation dans les prédictions :
- Consensus : `on` meilleur AUC isolé **0,645488**, IC95 bootstrap pistes **[0,544087, 0,748734]**.
- Adaptive : `on` **0,649985**, IC95 **[0,532914, 0,751871]**.
- Poly-gap : `off` **0,610919**, IC95 **[0,494773, 0,708808]**.

Ces valeurs AUC montrent une discrimination limitée, donc **aucun seuil nouveau ne peut être proclamé gagnant**. Les six nombres de chaque variante sont issus des événements MIDI256 réellement émis, pas de labels-oracles.

## Comparaison à S708

S708 reste la **seule référence officielle**, non modifiée ni re-promue. Les six scores CAL32 restent inchangés. Les cinq têtes spécialisées gelées sont indépendantes, et leur routeur S797 est dynamique à la frame et à la hauteur ; S803 **n'a pas créé une nouvelle architecture indépendante pour chacune des six métriques**. Aucune donnée CAL32/HELD/TEST n'a servi à sélectionner les politiques. Aucune mutation du code LIVE, horodatage de NoteOn/NoteOff source, délai d'offset, confirmation, lookahead ou buffer supplémentaire. **Latence physique du microphone non mesurée** : une causalité prouvée par replay n'est pas une preuve de zéro surcoût CPU sur l'appareil réel.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | [Run #39361632654](https://github.com/Andriamarosoa/note/actions/runs/39361632654), `S803_AUDIT_data` |
| Acoustique | PASS | [Run #39361632654](https://github.com/Andriamarosoa/note/actions/runs/39361632654), `S803_AUDIT_acoustics` |
| Architecture | PASS | [Run #39361632654](https://github.com/Andriamarosoa/note/actions/runs/39361632654), `S803_AUDIT_architecture` |
| Apprentissage | PASS | [Run #39361632654](https://github.com/Andriamarosoa/note/actions/runs/39361632654), `S803_AUDIT_training` |
| Métriques | PASS | [Run #39361632654](https://github.com/Andriamarosoa/note/actions/runs/39361632654), `S803_AUDIT_metrics` |
| LIVE | PASS | [Run #39361632654](https://github.com/Andriamarosoa/note/actions/runs/39361632654), `S803_AUDIT_live` |

**NVIDIA** : réponse initiale sur l'ancien run oui, mais première tentative sur le nouveau run **FAIL** (aucune réponse reçue), seconde tentative **en attente de preuve**. **Consensus 7/7** : pas encore attesté. Ne jamais fusionner des preuves appartenant à deux runs différents pour fabriquer un 7/7.

## Verdict

**Scientifique : trois hypothèses rejetées, aucun gain admissible, aucun remplacement de S708.** **Qualité des données + logiciel : six audits indépendants PASS sur le deuxième run.** La clôture documentaire 7/7 reste conditionnelle à la réussite réellement archivée de NVIDIA et du consensus sur ce même run. Les tests n'établissent pas de latence de microphone, ni six architectures complètes séparées par métrique.

## Prochaine action

Vérifier l'artefact NVIDIA et le consensus du run #39361632654 après la relance ciblée des seuls jobs en erreur. Pour la série suivante : diagnostic joint exact des attaques **et** extinctions (64 attaques vraies en collision S802 pouvant avoir un Offset faux), regroupement des faux positifs entre politiques, et recherche de caractéristiques par hauteur et fragment avec vraie contre-épreuve de chaque composition ; ne jamais retarder un NoteOff, ni utiliser les étiquettes de vérité en inférence.
