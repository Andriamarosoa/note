---
serie: S847
statut: en_cours
type: experience
reference: S708
branche: codex/note-s847-real-recorded-transposition256
run_id: 40094119273
commit_sha: 2cf7d4670abca5d7d8c8070700f5fe6dfc6766ce
promotion: false
---

# S847 — Transposition IA par sélection de vrais sons de guitare enregistrés

## Objectif
Après l'audit S846 ayant **rejeté** la fausse démo sonore S845 (médiane originale 199 Hz, sortie IA 11714 Hz, 56,5 % énergie >10kHz), construire une tête radicalement plus vérifiable : la décision de hauteur transposée est prise par une **IA entraînée** sur les vrais PCM256 GuitarSet, mais **le son rendu est directement issu d'un vrai enregistrement de corde joué physiquement**, jamais généré par un oscillateur, DSP pitch-shift, phase vocoder ou décodeur latent acoustique non supervisé.

Il s'agit honnêtement d'une **sélection neuronale de vrais échantillons instrumentaux**, pas encore d'une transposition fidèle de la performance ni d'une reconstruction acoustique de la prise originale. Il peut y avoir des changements de dynamique, d'attaque et de timbre. Le modèle conditionné par +12 et −12 doit apprendre à choisir les notes enregistrées correspondant à la transformation souhaitée ; aucun décalage de fréquence ne se calcule à l'inférence.

## Hypothèse
Le modèle S845 créait du bruit aigu car sa sortie audio de transposition latente manquait de cible acoustique transposée. Une nouvelle architecture peut séparer les risques : apprendre la **hauteur transposée** à partir du son micro, puis récupérer l'onde réelle du modèle à partir d'une banque de vraies notes hex GuitarSet. La qualité acoustique de chaque note source est alors physique par construction ; la validité musicale de la sélection, la continuité temporelle et la correspondance avec la performance restent à prouver.

## Modifications
- `src/onset256/real_recorded_transposer256.py` : encodeur CNN raw PCM256 + GRU des blocs précédents, conditionnement par shift embedding neuronal (−12, 0, +12), transporteur latent et tête cible MIDI40..88 entraînée, **sans calcul de fréquence**. Sortie `physical_recorded_wave256` est la copie directe de 256 valeurs d'un WAV de guitare réellement enregistré. Le système ne conserve aucune file d'audio futur du musicien et aucune sortie MIDI.
- `scripts/s847_train_real_recorded_transposer256.py` : extrait une banque de notes réelles de `audio_hex-pickup_debleeded.zip` et des annotations JAMS certifiées, uniquement guitariste TRAIN 04. Apprend le transpositeur neural sur vrais blocs microphone/GT (TRAIN 04, CAL 05, OUTER 00..03), sorties évaluées pour −12, 0, +12. Les labels de notes transposées sont **produits hors ligne** à partir de la vérité terrain, jamais utilisés à l'inférence. Les classes non observées restent inconnues, aucune note inventée. Les extraits hex sont de vraies captures, bien que l'ordre JAMS↔hex ait encore besoin d'un audit acoustique.
- `tests/test_s847_real_recorded_transposer256.py` : causalité, refuse les blocs >256, rétropropagation du réseau de shift, sortie audio bit-identique à la banque réelle, zéro MIDI, sélection de seuil sur CAL et OUTER isolé.
- Hérite des contrôles S846 rejetant les WAV artificiellement transposés aigus qui avaient trompé les premiers tests.

## Protocole
1. Télécharger les trois archives GuitarSet originales et vérifier intégralement SHA/MD5. Former une banque de **vrais enregistrements hex de notes**, annotée avec fichier/corde, exclure les notes trop faibles/silencieuses. Ne jamais fabriquer de clip ni utiliser de note sinusoïdale comme son de démonstration.
2. Modèle : à chaque frontière 256 samples de micro reçu, GRU passée autorisée, produire immédiatement probabilities de notes pour la vue ±12 sans formule de note à l'inférence. Restreindre la banque et les exemples TRAIN à guitariste 04 pour éviter fuite sur OUTER.
3. TRAIN/CAL/OUTER disjoints 04/05/00–03. Entraîner sur étiquettes réelles ; mesurer TP/FP/FN E2→E3 pour +12, E3→E2 pour −12 et identité. Sélection des seuils sur CAL uniquement.
4. Ne confondre **aucun** score de cible de note active avec une transposition audible correcte. Évaluer la qualité perceptive de la note physique, mais aussi la justesse des notes choisies, la continuité sans claquement à chaque changement, et l'audio de vraie guitare.
5. Un échantillon réel choisi au mauvais moment ne suit pas l'attaque, le vibrato et la dynamique du musicien. Il n'est donc pas une « vraie performance transposée ». Toute démo sera intitulée « vraie note enregistrée sélectionnée par IA », pas « guitare originale transposée ». Ne pas promouvoir si la précision de note ne dépasse pas le témoin de sélection ou si les discontinuités sont trop importantes.
6. Aucun effet sur Onset, Offset, Frame Pitch, Full Note, Exact-K, Active Exact-K et pipeline MIDI officiel. Rejouer 80 WAV et comparer Onset F1 >69,2584331 % avant toute intégration. Le code est à cadence 256 ; le p99 CPU Windows/iRig reste à prouver.

## Audits obligatoires
| Domaine | Statut requis |
|---|---|
| Données | vrai hex+micro/JAMS signé, banque de notes physiques captées, clips source TRACK TRAIN |
| Acoustique | audio du résultat = 256 échantillons copiés des véritables captures, aucune synthèse ni décalage de fréquence |
| Architecture | prédiction transposée par réseau conditionné ±12 et non une simple translation de bins |
| Apprentissage | GT source et cible calculés pour supervision HORS ligne, TRAIN/CAL/OUTER séparés |
| Métriques | F1 de notes actives et faux positifs en OUTER, audit du timbre et de la continuité |
| LIVE | entrée exactement PCM256, état du passé seulement, aucune attente ni MIDI |

## Statut
En développement, non promu. L'ancien S845 n'avait que **14,31 % F1 E2, 17,22 % E3, 10,05 % E2+E3**, et sa démo sonore est invalidée ; ce ne sont PAS des références valables d'une transposition audio fidèle. Onset PCM256 reste **69,2584331 % F1**.

## Traçabilité du run de recherche
[GitHub Actions S847 #40094119273](https://github.com/Andriamarosoa/note/actions/runs/40094119273). Cette expérience vérifie l'intégrité des sources mic/hex/JAMS avant la banque de vrais clips, l'entraînement et le test extérieur. À la dernière vérification, le téléchargement des archives officielles n'était pas terminé. **Aucun nouveau score ni démo S847 fidèle ne sont revendiqués avant résultat validé.** Tests de conformité PCM256 antérieurs à la phase de téléchargement effectués sans échec.
