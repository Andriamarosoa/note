# Résultat — Onset compilé PCM256

## Parité sur l'audio complet

Validation locale du 23 octobre 2026 avec Python 3.12 et les versions de
`requirements-onset256.txt` :

- **80 WAV**, **306536 blocs de 256**, **12276 attaques émises**.
- Chaque paire `(sample, pitch)` est identique aux prédictions scellées du
  [run 39915038301](https://github.com/Andriamarosoa/note/actions/runs/39915038301).
- **TP 8012 / FP 4264 / FN 3255 / F1 68,062694 %**.
- Aucun réentraînement ni changement de seuil pendant l'optimisation.
- Aucun bloc futur, aucune attente ajoutée et aucune rétrodatation.
- 79 tests validés ; les 3 tests de compilation ont aussi été rejoués après
  l'ajout des gardes de dimension. Les 31 empreintes des fichiers préservés
  et le chargement du bundle sur dix blocs consécutifs ont été vérifiés.

Le rapport intégral est dans `evidence/compiled_replay_report.json`.
Le replay a utilisé cinq processus pour vérifier la parité. Ses temps ne
servent pas de benchmark de latence matérielle.

## Mesures CPU avec le nouveau HGB

Deux passages séquentiels sur le même préfixe `00_BN1-129-Eb_comp` :
1000 blocs chacun, les 100 premiers exclus des mesures, **26 attaques
exactement identiques**. Compilation et chargement avant l'entrée audio.
Aucune autre tâche de replay n'était lancée en parallèle.

| Ordre | Chemin | p50 | p95 | p99 | Blocs > 5,805 ms |
|---|---|---:|---:|---:|---:|
| Référence puis compilé | Référence | 8,312 ms | 16,445 ms | 21,775 ms | 895/900 |
| Référence puis compilé | Compilé | 6,979 ms | 20,513 ms | 53,009 ms | 566/900 |
| Compilé puis référence | Compilé | 7,204 ms | 15,335 ms | 21,872 ms | 609/900 |
| Compilé puis référence | Référence | 13,645 ms | 24,236 ms | 30,397 ms | 900/900 |

Le second ordre a été testé après les dépassements observés dans le premier.
Tous les résultats sont conservés, y compris la régression de p95 du premier
passage. Le second passage mesure aussi le temps CPU du processus : p95
compilé **15,306 ms**, proche du temps écoulé. On ne peut donc pas attribuer
les dépassements à la seule désallocation du processeur par le système.

**Le budget de calcul de 5,805 ms n'est pas tenu dans ces mesures.**
La sortie est causale par blocs de 256, mais le temps réel sans dépassement
n'est pas validé. La baisse du temps médian ne suffit pas à autoriser une
telle conclusion. Aucun microphone ni périphérique MIDI n'a été mesuré.

Le premier benchmark, antérieur et utilisant le contrôle69, avait obtenu
p95 5,242 ms et p99 7,027 ms pour le chemin compilé. Il est conservé séparément
dans `control69_prefix_benchmark.json` ; ces chiffres ne sont pas ceux du
nouveau HGB et ne certifient pas son budget CPU.

## Reproduction

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.benchmark_compiled_onset256 \
  --root DONNEES_SIGNEES --previous RESULTATS_RUN_39915038301 \
  --out benchmark.json
```

Ajouter `--reverse` pour inverser l'ordre. Les deux rapports du nouveau HGB
sont dans `evidence/hgb_prefix_benchmark_forward.json` et
`evidence/hgb_prefix_benchmark_reverse.json`.

Les modèles restent des poids de recherche FIT80. La parité d'exécution
ne constitue ni une validation externe, ni une promotion du MIDI officiel.
