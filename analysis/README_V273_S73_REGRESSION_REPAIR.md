# S73 — Audit des régressions et politique de correction (11 octobre 2026)

**Statut : non promu.** Les balayages S73 ont produit **8 400 + 11 220 + 3 360 = 22 980 configurations**, enregistrées avec leurs empreintes SHA-256 dans `analysis/v273_s73_repair_manifest.json`. Les prédictions individuelles des sélections S73 historiques **n'ont pas été conservées** : un Exact-K par K0–K6 ne peut pas être reconstitué honnêtement à partir de leurs agrégats.

| Politique | Global Exact-K | Poly Exact-K | Actions (étape 1) | Fixes étape 1 | Régressions étape 1 | Net global vs S72 | Net poly vs S72 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **S72** | **49 238 / 59 309 (83,0194%)** | **3 202 / 7 385 (43,3582%)** | – | – | – | – | – |
| S73 permissif (deux sélections) | 48 581 (81,9117%) | 3 622 (49,0454%) | 1 917 | 379 | **1 084** | **−657** | +420 |
| S73 max-poly avec gain global | 49 271 (83,0751%) | 3 334 (45,1456%) | 2 321 | 830 | 797 | +33 | +132 |
| **S73 conservateur** | **49 298 (83,1206%)** | **3 330 (45,0914%)** | **1 629** | **587** | **527** | **+60** | **+128** |

**Défaut identifié :** le premier étage permissif effectue 1 917 actions pour 379 corrections et **1 084 régressions** ; il fait perdre 705 décisions globales nettes en échange de +372 poly. Son deuxième étage apporte +48 global/poly (216 corrections/168 régressions, 575 actions), insuffisant pour compenser. Les taux sont mesurés sur les folds historiques exposés 0/1/2/4 et la variante conservatrice a été **choisie a posteriori** dans la grille.

**Sélection corrective à conserver pour rejeu (pas pour promotion) :** `poly_encouraged`, poids `1.6`, domaine `all`, gate `relative`, seuil `0.20`. Parmi les variantes avec régressions ≤600, gain global strictement positif et gain poly positif sur chaque fold, elle atteint **45,09% poly /83,12% global**, +20/+36/+29/+43 poly par fold. Elle évite **270 régressions** par rapport au max-poly équilibré pour seulement quatre accords poly corrects nets en moins. Ce n'est pas une amélioration prouvée sur du neuf.

**Nouvelles têtes testées (OOF, acoustique 58 + géométrie 46 + temps 392 + probas PR16 84 + votes) :**
- filtre K0/K1 vers K2–K6 : 8 actions ; 3 fixes K2, 3 régressions K1 ; **0 net global, +3 poly** ; rejeté.
- filtre K2→3 / K3→4 / K4→5 / K5→6 : 10 actions ; 2 fixes, 3 régressions ; **−1 global/poly** ; rejeté.

**Correction du protocole :** `scripts/audit_v273_s73_candidate.py` vérifie les 59 309 mêmes IDs et y, les chiffres natifs S72 et, **uniquement si toutes les prédictions S73 existent et reproduisent l'agrégat déclaré**, calcule les 7 Exact-K, folds, transitions, corrections/régressions. Sans ce fichier, statut `BLOCKED_MISSING_S73_CASE_PREDICTIONS`. Le workflow CI réussit en confirmant ce blocage, mais **n'annonce pas une validation ou promotion**.

**Référence officielle inchangée :** `freeze_local_combo`. **Référence de recherche inchangée :** S72. S73 conservateur est seulement la nouvelle piste de rejeu, sous réserve de retrouver les prédictions individuelles et d'obtenir un véritable nouveau test.