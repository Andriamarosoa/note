---
serie: S846
statut: en_cours
type: audit
reference: S708
branche: codex/note-s846-transpose-audio-fidelity-gate256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S846 — Audit réel : rejeter toute fausse démo transposée aiguë non guitare

## Objectif
Répondre à la plainte légitime : la démo sonore S845 diffusée a été perçue comme uniquement un son aigu, sans vraie guitare transposée. L'ancien test GitHub vérifiait seulement la présence de signal numérique, un exemple à 44100Hz, un checkpoint existant, et une amplitude non nulle. Ce n'est pas une certification de qualité sonore ni une confirmation de transposition à +12 demi-tons.

## Mesure effectuée sur LE FICHIER WAV réellement communiqué
**Guitare originale** : centroïde spectral 621,58Hz, médiane 199,18Hz, 1,57% énergie >10kHz, 72,87% énergie 80–250Hz.
**Partie sortie IA +12** : centroïde 10979,65Hz, médiane 11714,06Hz, **56,5% énergie >10kHz**, seulement 1,92% énergie 80–250Hz.
Malgré le volume RMS final normalisé à ~0,18 dans les deux sections, le vrai échec est l'emplacement de l'énergie et le peu de dynamique. L'ancien exportateur avait normalisé le RAW IA à gain ×15,64 et déclaré PASS parce que non zéro.

## Hypothèse
Le décodeur audio S845 des vues latentes ±12 ne produit pas réellement un événement musical d'octave. Ce qui est produit est une estimation sans entraînement suffisamment physique pour la forme d'onde transposée, alors même que la reconstruction des six canaux hex non transposés battait moins le silence en MAE (0,029422 vs 0,029205). La cause précise du bruit aigu peut comprendre l'absence de supervision de la forme d'onde latente transposée ou l'écrasement du décodeur, à prouver, et pas un défaut d'écoute de l'utilisateur.

## Modifications
- `scripts/s845_make_real_audio_ab_demo256.py` : garde `audit_transposed_audio` décomposant l'énergie originale/IA, médiane/centroïde spectral, pourcentage >10kHz et évolution de dynamique. Lève **ValueError et ne crée aucun WAV** quand une guitare grave se transforme presque uniquement en son très aigu et stationnaire.
- `tests/test_s846_transposition_audio_fidelity256.py` : tests logiciels fictifs uniquement, ne constituent jamais une démo ni une évaluation musicale.
- Workflow GitHub `note-s846-transposed-audio-gate256.yml` télécharge la vraie démo S845 originale run 40091786843 et vérifie **que ce fichier précis est rejeté** par le garde. Une CI PASS signifie seulement que ce faux positif n'est plus accepté. Elle n'établit pas que S845 sait transposer la guitare.

## Protocole
Ne pas résoudre par amplification, filtrage passe-bas arbitraire, resampling, oscillateur, son synthétique ou fausse démo. Priorité à l'audit physique micro/hex, à des signaux transposés réellement joués et enregistrés sur la guitare, et à la supervision de la forme d'onde **de la vue transposée**, pas seulement des étiquettes MIDI. Pour autoriser une démo, mesurer les vrais spectres, l'enveloppe, la hauteur fondamentale attendue et la reconstruction selon des enregistrements distincts validés. Les garde-fous de bruit ne prouvent pas une transposition +12 correcte : une simple copie de l'original pourrait passer. Exiger l'audit des hauteurs sans truquer le résultat.
Tout reste causal avec exactement 256 nouveaux échantillons par décision, sans lookahead ni attente, indépendant des autres caractéristiques, zéro changement MIDI.

## Audits obligatoires
| Domaine | Résultat |
|---|---|
| Données | vrai fichier S845 téléchargé et comparé à la source, aucune nouvelle archive synthétique |
| Acoustique | médiane spectrale et énergie aiguë, enveloppe de volume, aucune fausse transposition acceptée |
| Architecture | mêmes poids S845 intacts ; nouvelle validation hors architecture |
| Apprentissage | aucun nouveau modèle entraîné, score Onset inchangé |
| Métriques | contraste d'énergie et qualité sonore, non F1 Onset |
| LIVE | 256 / sans futur, aucun MIDI ni déploiement, p99 matériel non mesuré |

## Statut
Démo S845 **INVALIDÉE**. Une sortie audio estimée par l'IA n'est pas automatiquement un vrai son de guitare transposé, même lorsqu'une source originale était réelle. Aucune nouvelle démo fiable ne peut être annoncée avant une nouvelle méthode réussie et un test d'écoute et de référence. Témoin Onset **69,2584331 % F1** intact.
