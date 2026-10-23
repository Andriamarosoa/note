---
serie: S836
statut: en_cours
type: intervention_profiling_live
reference: S835/S834
branche: codex/note-s836-live-sharp-cache256
promotion: false
---

# S836 — Première boucle mesurable d'amélioration du vrai LIVE NOTE

## But et protocole
Maintenir le live NOTE jouable S613/S614 intact, prendre le registre des six meilleures têtes conservées en S835 sans les promouvoir, et enlever les surcoûts CPU de la voie SHADOW PCM256 S834 pour préparer une future intégration sans latence additionnelle. Le contrôle est toujours la **même** série GuitarSet et le **même** modèle S708, sans retoucher des poids ou un sélecteur musical.

## Contrat non négociable
- 44 100 Hz, exactement 256 échantillons nouveaux par décision (5,80498866 ms) ; zéro futur PCM, zéro confirmation ajoutée, zéro attente supplémentaire.
- Six axes **indépendants** : Exact-K, Active Exact-K, Onset, Offset, Frame Pitch, Full Note. Chaque axe conserve HEAD/SELECTOR/DECODER/état privés. Aucun transfert des sorties d'une tête comme entrée d'une autre.
- Les références H0, S828, S834, S835, le moteur microphone MIDI et les modèles originaux ne sont **pas modifiés**.
- Référence SHADOW S834 : Exact-K 75,780659 %, Active-K 65,193596 %, Onset F1 71,382285 %, Offset F1 38,677437 %, FramePitch F1 84,998850 %, FullNote F1 71,108037 % sur 80 pistes FIT80. Onset F1 < H0 73,252019 % ; ne pas promouvoir. Ces scores ne forment pas un unique MIDI.
- Pas de score mixte : comparer F1 à F1, accuracy à accuracy, FIT80 à FIT80, CAL32 à CAL32, shadow à shadow. Matériel Windows, qualité Live et six-axes unifiés non certifiés.
- Toute tentative donnant un faible gain ou une régression doit être **auditée causalement** : exemples, changement concret, TP récupérés/perdus, FP ajoutés/retirés, splits TRAIN/CAL/OUTER, budget CPU. Ne pas changer aussitôt de méthode sans comprendre.
- La modification n'ajoute aucun lookahead, ne touche aucune autre tête et ne modifie pas le MIDI public.

## Hypothèse falsifiable H1 — FFT S706
Dans S834, à chaque hop 256, `sharp_track(audio, [n-256, n])` calcule **deux** fois les 3 FFT Hann de 2048/4096/8192 pour produire la différence de phase. L'analyse du hop précédent est identique à celle déjà calculée au hop précédent. Une mise en cache des amplitudes complexes *passées uniquement* peut diviser approximativement par deux le travail spectral, tout en gardant exactement les 96 valeurs de l'ABI S706.

**Intervention isolée** :
- `src/causal_note/s836_incremental_sharp256.py` : cache des spectres complexes interpolés précédents et des constantes, strictement un nouvel échantillonnage FFT par longueur à chaque bloc.
- `src/causal_note/s834_s809_native256_audio.py` : `sharp_engine="incremental"` **optionnel**, défaut `reference`, donc comportement historique inchangé tant que le contrôle n'est pas demandé.
- `test/test_s836_incremental_sharp256.py` : comparaisons composante par composante, attaques, silence, rolling 8192, mutation audio futur, six décodeurs privés et horodatages.
- `scripts/s836_sharp_cpu_benchmark.py` : même signal synthétique, p50/p95/p99 référence contre cache ; n'évalue pas la musique.

## Gardes de promotion
(1) Parité des 96 caractéristiques **bit pour bit** (strict `np.array_equal`), (2) parité de chaque six-axis décision et état sur mêmes données, (3) mesures CPU sur vraies pistes avec les poids figés, (4) pas de régression de score dans les six axes, (5) CPU complet p95/p99 inférieur à 5,805ms dans un test ciblé et mesures de latence matérielle iRig avant toute prétention de meilleur LIVE. Les conditions restantes sont BLOQUANTES, pas « optionnelles ».

Les agents NVIDIA externes personnalisés de Work n'ont **pas été invoqués** dans cette série : ne pas présenter les tests CI comme leurs audits.
