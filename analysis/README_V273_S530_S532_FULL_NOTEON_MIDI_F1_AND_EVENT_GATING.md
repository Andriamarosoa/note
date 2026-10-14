# S530–S532 — NoteOn événementiel complet MAESTRO et audit des fausses naissances

**15 octobre 2026 — travaux de développement sur données déjà connues, AUCUNE PROMOTION.**

Les étapes S525–S529 amélioraient des **accords complets conditionnels** sur les 322 onsets MIDI appariés à V88 et protégeaient les accords déjà corrects de Basic Pitch, mais laissaient de 71 à 170 émissions de notes supplémentaires sur des événements non appariés, sans métrique NoteOn intégrale.

## S530 — Evaluation temporelle avec chaque vraie note MIDI et chaque fausse prédiction

[CI S530 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38648729508). Les 3 extraits MAESTRO 12 s comportent **330 vrais groupes MIDI NOTE_ON** (416 hauteurs frappées au total). Les 575 candidats V88 d'origine sont tous considérés ; 322 vrais groupes avaient été appariés historiquement, et les 8 autres restent des faux négatifs. Les MIDI officiels sont relus et vérifiés SHA par rapport à S471 ; aucun MIDI ne sert à décider des notes. Les TP exigent la **bonne hauteur MIDI et un début à ±80 ms**, avec appariement 1:1 par hauteur. Tous les faux déclenchements sont comptés. **Ce n'est pas un test aveugle**, les enregistrements ont déjà servi aux choix de conception.

| Politique (sources S528/S529 gelées) | TP | FP | FN | Précision | Rappel | F1 NOTE_ON | F1 attaques (indépendant des hauteurs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naissances BP seules | 69 | 30 | 347 | 69,70 % | 16,59 % | 26,80 % | 36,98 % |
| Raw pitch p≥0,50, protéger BP | 273 | 598 | 143 | 31,34 % | 65,63 % | 42,42 % | 76,96 % |
| **Raw pitch p≥0,70, protéger BP** | **213** | **206** | **203** | **50,84 %** | **51,20 %** | **51,02 %** | **77,96 %** |
| Raw+BP p≥0,50, protéger BP | 267 | 626 | 149 | 29,90 % | 64,18 % | 40,79 % | 75,34 % |
| Raw+BP p≥0,70, protéger BP | 212 | 232 | 204 | 47,75 % | 50,96 % | 49,30 % | 76,34 % |

Le score de 51,02 % est une mesure **NOTE_ON pitch+onset** sur toutes les attaques MIDI de cette cohorte, et non le F1 « hauteur conditionnel sur un onset connu » de S528/S529. Il est obtenu avec un modèle réellement entraîné sur 2 autres enregistrements parmi ces 3, mais **la cohorte a été explorée auparavant**, donc aucune démonstration de performance générale. Source K = V88 piano partiel, **pas** le S80 complet.

## S531 — Deux architectures de validation d'attaque, 3 folds tenus à part

[CI S531 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38649040041). Apprentissage séparé sur deux morceaux et test sur le troisième ; **6 vrais poids d'experts** ExtraTrees / LogisticRegression. Les 322 candidats V88 appariés à un vrai onset MIDI sont la classe positive. Les **253 candidats V88 non appariés** sont utilisés comme négatifs **selon la règle 1:1 ±80ms du benchmark** ; cela inclut des doublons proches de vrais onsets, et **ne prouve pas 253 événements acoustiquement silencieux**. Les caractéristiques d'entraînement sont uniquement les 49×11 caractéristiques audio S527 agrégées, V88 K souple, indices prélabel BP et voisinage temporel ; aucune probabilité issue de l'entraînement S528 sur le morceau tenu à part ne sert comme caractéristique du filtre S531.

Le sélecteur laisse intacts les BP notes déjà présentes et n'autorise l'émission de nouvelles notes qu'à la condition que le validateur onset accepte l'événement.

| Gate + seuil | TP | FP | FN | F1 NOTE_ON |
|---|---:|---:|---:|---:|
| Aucun, S530 raw p≥.70 | 213 | 206 | 203 | **51,02 %** |
| ExtraTrees onset p≥.50 | 158 | 114 | 258 | 45,93 % |
| ExtraTrees onset p≥.70 | 126 | 71 | 290 | 41,11 % |
| Logistique onset p≥.50 | 177 | 141 | 239 | 48,23 % |
| Logistique onset p≥.70 | 137 | 100 | 279 | 41,96 % |

Les six têtes ont réellement appris. Malgré une meilleure précision, elles perdent **plus de vrais onsets** qu'il n'est utile pour le F1. **Aucune amélioration ni promotion** ; les gates trop agressifs doivent être rejetés.

## S532 — Limiter séparément le nombre de notes nouvelles

[CI S532 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38649339556). Test sans nouveau modèle : conserver les notes déjà nées dans BP ; seulement sur les événements BP initialement vides, garder les notes du réseau S528 raw p≥.70, limitées par rang aux 1, 2, 3 meilleures. Ne jamais utiliser le K0 prédit comme plafond strict.

| Nombre maximal de nouvelles notes | TP | FP | FN | F1 NOTE_ON | Accords complets correctement identifiés |
|---|---:|---:|---:|---:|---:|
| Illimité, S530 vérifié | 213 | 206 | 203 | **51,02 %** | 80 |
| 1 | 176 | 138 | 240 | 48,22 % | 97 |
| 2 | 204 | 182 | 212 | 50,87 % | 80 |
| 3 | 210 | 196 | 206 | **51,09 %** | 80 |

Le cap 3 gagne **0,08 point de pourcentage**, quantité négligeable et issue d'une cohorte historiquement explorée. Même le cap1 qui améliore l'identité d'accords entiers **dégrade le F1 NOTE_ON total**. Aucune promotion S532.

## Diagnostic et suite scientifique

1. Les **vraies attaques** ne se déduisent ni de K prédit, ni du fait qu'un candidat n'a pas de note BP. La suppression K0 qui détruisait 102 attaques S526 est corrigée **en développement seulement** par la présence de 575 centres V88 K0..K6.
2. Les propositions de hauteur confondent encore des partiels harmoniques avec des fondamentales jouées, particulièrement lorsque la note présente est faible. Un simple filtre onset binaire ne résout pas ce problème.
3. Les suppressions apprises sur 2 anciens morceaux généralisent mal au troisième ; le classement topN constitue un correctif trop faible. Il faut une **tête note-vs-harmonique couplée à un modèle frame+onset multi-temporel**, plutôt qu'une addition de seuils fondés sur les mêmes erreurs.
4. Il est nécessaire d'entraîner sur **plusieurs nouveaux morceaux/compositeurs** avec des annotations MIDI complètes du flux, et de conserver un **jeu final complètement scellé** : aucun choix de seuil/architecture n'est fait sur le jeu de test final.
5. Les vrais métriques de promotion devront être F1 NoteOn à ±80 ms par hauteur, faux déclenchements, Exact-K K0–K6, accords exacts, offset/durée et latence Live. Évaluer séparément guitare IDMT, MAESTRO piano et GuitarSet **à protocole comparable**. Pour le moment la chaîne S80 complète n'a pas été exécutée sur MAESTRO.

**Règle : conserver comme candidat diagnostique S530 p≥.70 (51,02 % NoteOn sur le corpus historique) et non comme référence officielle. S531 rejeté. S532 ≤0,08 pt net exploratoire, rejeté. Aucun changement S80, V88, freeze_local_combo, Live ou onsets/offsets de production.**
