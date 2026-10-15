# S561–S568 — 18 compositeurs, nouvelle tête physique et troisième validation NoteOn sans contamination du test

**15 octobre 2026.** Les sorties sont liées aux GitHub Actions réussis, avec des scripts et des SHA enregistrés sur la branche de [PR #41](https://github.com/Andriamarosoa/note/pull/41). **Aucune promotion, fusion ni modification S80 Exact-K, freeze_local_combo, GuitarSet, IDMT ou Live.**

## 1. Constitution d'une cohorte réellement non entraînée

[S561 SUCCESS #38666895837](https://github.com/Andriamarosoa/note/actions/runs/38666895837) : sélection uniquement sur le CSV officiel MAESTRO v3, partition **originale TRAIN** (ce n'est pas le score du split officiel TEST), de cinq morceaux nouveaux dont les noms de compositeurs/co-compositeurs ne figurent pas dans les **18 précédents enregistrements** déjà utilisés par S469/S537/S545/S554. Cinq œuvres : Johannes Brahms, Henry Purcell, Isaac Albéniz, Nikolai Medtner, Mikhail Glinka / Mily Balakirev. Toutes sélectionnées par hachage déterministe des seuls chemins d'audio avant lecture de leur WAV/MIDI. L'absence d'exposition dans tout l'historique des autres branches GitHub n'a pas été vérifiée exhaustivement. Il s'agit strictement de cinq nouvelles sources exclues de l'entraînement S546 et S562, avec véritable gel prédiction avant étiquettes.

## 2. Apprentissage physique sur 18 sources (S562)

[S562 SUCCESS #38667229236](https://github.com/Andriamarosoa/note/actions/runs/38667229236), après une correction triviale du nom des caractéristiques au selftest. **18 enregistrements réels** déjà annotés (3+5+5+5) pour entraînement, avec exclusion de tous les morceaux partageant un co-compositeur avec celui tenu à part dans chaque pli. Deux experts appris avec de véritables modèles ExtraTrees (et deux checkpoints finaux complets), en gardant les classes MIDI 40..88, les candidats V88 y compris K0 et les BP births protégées :

- **Raw7** : 7 descripteurs physiques originaux de chaque note (hauteur normalisée, énergie, force, attaque, maintien, nouveauté, indicateur physique).
- **Competing-fundamental34** : ces 7 descripteurs + rapports de concurrence de fondamentales plus graves aux intervalles 12, 19 et 24 demi-tons ; différences d'énergie/attaque/nouveauté, voisinage de hauteur, enveloppe maintien vs attaque. 34 descripteurs réellement calculés depuis les WAV, pas de vérité MIDI dans le vecteur.

Résultats **rétrospectifs de validation croisée interne sur les 18 enregistrements**, ±80ms et note MIDI exacte : **Raw7 TP1452 / FP1371 / FN924, F1 55,86 %** ; **Physique34 TP1342 / FP841 / FN1034, F1 58,87 %**. Gain physique34 = +3,02 pp en F1, 530 faux positifs éliminés, mais 110 notes vraies supplémentaires manquées. Ce n'est **pas** un résultat prospectif sur une source nouvelle ni une garantie de non-régression.

## 3. Vraie évaluation indépendante sur de nouveaux WAV (S563–S567)

Tous les scripts se terminent sur des workflows PASS à la date ci-dessus :

1. [S563 #38667195568](https://github.com/Andriamarosoa/note/actions/runs/38667195568) : vrai son WAV de cinq nouveaux extraits 12s, candidats acoustiques V88 et poids freeze_local_combo originaux, aucune vérité MIDI consultée, source SHA.
2. [S564 #38667294210](https://github.com/Andriamarosoa/note/actions/runs/38667294210) : même modèle préentraîné Basic Pitch, notes MIDI40..88 et empreintes SHA figées, aucune annotation.
3. [S565 #38667773991](https://github.com/Andriamarosoa/note/actions/runs/38667773991) : original 49x11 vraies caractéristiques acoustiques MIDI-blind à tous les événements V88, y compris K0. Les premiers workflows avaient une dépendance pointant vers un ancien run ; ceci a été corrigé AVANT toute lecture d'annotation.
4. [S566 #38668099481](https://github.com/Andriamarosoa/note/actions/runs/38668099481) : chargement et contrôle SHA des huit modèles S546 originaux et des deux poids réellement entraînés complets S562, inférence sur les **cinq nouveaux WAVs**, seuil de pitch 0,70 inchangé et sept politiques prédéclarées : BP seulement, S546 dedup80/dedup100+harmonic, S562 raw7 dedup80/dedup100+harmonic, S562 physical34 dedup80/dedup100+harmonic. Tous les fichiers de notes prédites MIDI40..88 et leurs SHA sont conservés **avant la lecture de l'un quelconque des cinq MIDI**.
5. [S567 #38668197551](https://github.com/Andriamarosoa/note/actions/runs/38668197551) : seulement **APRÈS** ce gel, téléchargement des cinq MIDI officiels et vérification de leur source. Appariement pitch MIDI exact à ±30/50/80ms, tous les FP et FN comptés, les cinq vrais extraits comportent **310 notes MIDI40..88**, aucune réadaptation d'entraînement/seuil sur ces annotations.

### S567 : scores prospectifs réellement obtenus

| Politique figée | TP | FP | FN | Précision | Rappel | F1 NoteOn ±80ms |
|---|---:|---:|---:|---:|---:|---:|
| Basic Pitch naissances seules | 61 | 35 | 249 | 63,54 % | 19,68 % | 30,05 % |
| S546 originale + dedup80 | 205 | 233 | 105 | 46,80 % | 66,13 % | 54,81 % |
| **S546 originale + dedup100/harmonic** | **201** | **212** | **109** | **48,67 %** | **64,84 %** | **55,60 %** |
| S562 raw7 full18 + dedup80 | 231 | 364 | 79 | 38,82 % | 74,52 % | 51,05 % |
| S562 raw7 full18 + dedup100/harmonic | 222 | 330 | 88 | 40,22 % | 71,61 % | 51,51 % |
| S562 physical34 full18 + dedup80 | 204 | 214 | 106 | 48,80 % | 65,81 % | 56,04 % |
| **S562 physical34 + dedup100/harmonic** | **201** | **202** | **109** | **49,88 %** | **64,84 %** | **56,38 %** |

Le nouveau modèle physique **gagne 0,78 point de F1 sur le même test** par rapport à S546 100ms/harmonic, **sans ajouter ni perdre de note correctement détectée dans ce corpus** (201 TP inchangés) : il retire 10 FP de plus. Au contraire, le modèle raw7 complet à 18 sources sur-prédit et F1 régresse malgré un meilleur rappel.

La comparaison S567 *par nouveau compositeur* physical34 vs S546 100ms/harmonic révèle les risques de régression : **Brahms −4,96 pp, Albéniz −2,17 pp**, Purcell +4,51 pp, Medtner +2,80 pp, Glinka/Balakirev +5,37 pp. Donc le gain de 0,78 pp n'est **pas systématique** et ne suffit pas à lui seul pour promouvoir une architecture.

## 4. Audit par registre sur troisième cohorte (S568)

[S568 PASS #38668327687](https://github.com/Andriamarosoa/note/actions/runs/38668327687) : relit uniquement les MIDI déjà utilisés dans S567, vérifie leur SHA, reproduit exactement tous les compteurs TP/FP/FN des sept politiques selon les trois registres et les trois tolérances. C'est un audit **après exposition des étiquettes**, jamais une autre mesure « unseen ».

Les 310 vraies notes se répartissent : MIDI40–59 = **129**, MIDI60–71 = **119**, MIDI72–88 = **62**.

| Registre | S562 physique34 100ms/harmonic vs S546 même filtrage | Variation TP | Variation FP | Différence F1 |
|---|---|---:|---:|---:|
| Grave (40–59) | Plus de notes correctes mais également plus de faux positifs | +11 | +31 | **+0,91 pp** |
| Médium (60–71) | Principal gain de réduction de faux harmoniques mais **8 vraies notes perdues** | −8 | −30 | **+2,46 pp** |
| Aigu (72–88) | 11 faux positifs filtrés, 3 vraies notes perdues | −3 | −11 | **+1,87 pp** |

Comparé au **nouveau physique34**, l'ancien raw7 18-source dégradé à 100ms/harmonic ajoute +45 FP pour +5 TP dans le grave, +48 FP pour +9 TP au médium, +35 FP pour +7 TP dans l'aigu : le compromis spectral supplémentaire permet de réduire des hallucinations de notes, mais le rappel des vraies réattaques et fondamentales reste insuffisant.

## 5. Décision technique

**Rien à promouvoir dans la référence officielle.** Le meilleur modèle testé sur la nouvelle cohorte S561 est la tête physique34 **56,38 %**, une amélioration modeste de **+0,78 pp** sur S546 à 55,60 % sur les mêmes cinq morceaux. Le meilleur score observé sur une AUTRE cohorte S554 était auparavant **60,02 %** avec S546, non comparable comme progression d'une même expérience ; ne pas transformer le contraste 60,02→56,38 en régression expérimentale imputée au modèle sans score S546 apparié.

**Prochain défaut à résoudre en entraînement, et non à la main sur S561 désormais exposé** : apprentissage distinct de l'attaque nouvelle et de la résonance existante, avec validation sans même source ; ajouter des labels explicites de retransitoires et vrais rejouements d'une même hauteur afin de protéger les 8 TP médiums perdus tout en rejetant les 30 faux positifs médiums. Évaluer ensuite sur une quatrième cohorte préenregistrée avant les MIDI. La simple combinaison/vote des anciennes têtes avait été rejetée en S553.

Source V88 original distinct du système S80 complet : ces runs ne rejouent PAS les anciens exact-K GuitarSet/IDMT et ne prouvent PAS une latence Live. Aucun composant officiel modifié, modèle expérimental conservé uniquement sur la branche de PR #41, et objectif **80 % de F1 NoteOn non atteint**.
