# S465–S468 — Séparer naissance de note et continuation sans toucher Exact-K

Date : 14 octobre 2026. Toutes les validations sont rétrospectives et **ne remplacent pas** la référence officielle S80, le détecteur Live, les onset/offset ou le décodeur S446 `son → noms de notes`.

## Problème établi, sans nouvelle interprétation des annotations

S462–S464 ont mesuré sur IDMT Dataset3 (cinq enregistrements × 10 s) : **47 attaques XML réelles regroupées, toutes détectées dans ±80 ms**, mais 506 événements/candidats V88 ; 418 sont à plus de 80 ms de chaque attaque XML, dont 417 sont **prédits** K0 par S80. La très grande majorité des K0 éloignés conservent une énergie non négligeable sans nouvel accroissement RMS franc, compatible avec la continuation de la vibration. Des candidats éloignés ne sont pas des « vrais négatifs » annotés : ne jamais leur attribuer implicitement un vrai K0.

Sur les 45 attaques dont les hauteurs sont dans MIDI40..88, **7 attaques réelles sont faussement étiquetées K0** par S80, dont 3 accords polyphoniques. Un filtre inconditionnel K0 est donc interdit.

## S465 — contrôle avant entraînement : une augmentation RMS ne suffit pas

[GitHub Actions S465 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38551837031).

Une grille uniquement *diagnostique* (-4,0,2,4,6,8,12 dB) a été calculée sur les 506 événements de S464 à partir des vrais signaux originaux, **sans déplacer un seuil de modèle**.

| Filtre contre-factuel | Candidats éloignés écartés / 418 | Vraies attaques conservées / 47 | Vrais accords poly conservés /21 |
|---|---:|---:|---:|
| Hausse RMS ≥ 0 dB | 376 | 41 | 18 |
| Hausse RMS ≥ 6 dB | 414 | 40 | 18 |
| Hausse RMS ≥ 6 dB + conservation des K prévus non nuls par S80 | 414 | 41 | 18 |

Le filtre « avec K non nul protégé » perd encore **6 vraies attaques dont 3 polyphoniques**, en particulier celles à tort classées K0 par S80. Ne pas promouvoir ces filtres malgré la baisse impressionnante du nombre de candidats.

## S466 — nouveau détecteur binaire *naissance vs non-naissance*, sans entrée K de S80

[GitHub Actions S466 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38552253544).

**Données d'apprentissage uniquement GuitarSet** : 59 309 exemples acoustiques natifs S258 avec vérité séparée par annotation JAMS (19 657 vrais événements avec note naissante, 39 652 candidats sans naissance annotée). 1 090 vraies caractéristiques acoustiques S97 : résumé58, flux245, morphologie490, physique294, géométrie3. L'architecture binaire ExtraTrees, quatre modèles / quatre folds de compositions disjointes. Sous-ensemble de validation interne **par autres compositions entières**, pour choisir un seuil de rétention imposant au moins 99,5 % de rappel des vraies naissances et des accords polyphoniques, ainsi que des vraies naissances que S80 avait classées K0. Le modèle ne reçoit jamais les K source en entrée.

Seuils gelés par fold : **0,075 ; 0,125 ; 0,075 ; 0,05**, issus exclusivement de GuitarSet. Le K source ne sert qu'en garde aval : **K original non nul est toujours conservé**. Sur les 59 309 exemples tenus à l'écart par composition, S466 propose d'écarter 2 140 candidats et perd **8 vraies attaques non polyphoniques**, **0 vraie attaque polyphonique**. Il n'est donc pas admissible comme filtre officiel zéro-régression malgré ses bons résultats de développement.

## S467 — score des cinq vrais WAV IDMT, figé avant les annotations

[GitHub Actions S467 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38552460959).

Restaure la véritable entrée 1 090D acoustique et les K S80 originaux des 506 événements IDMT, les quatre modèles S466 vérifiés par SHA, et leurs seuils GuitarSet déjà figés. Chaque modèle calcule sa probabilité de « nouvelle attaque », sans ouvrir aucun XML IDMT. On n'autorise une proposition d'écarter un candidat que si :

- son K original S80 est égal à zéro ;
- les **quatre** modèles ont effectivement calibré leur seuil sur GuitarSet ;
- les **quatre** modèles placent la confiance « attaque naissante » sous leur seuil GuitarSet.

Résultat gelé : **33 propositions d'écartement**, 473 événements conservés, **0 décision K de S80 modifiée**, 0 événement supprimé réellement dans la chaîne Live/S80. Les noms de notes S446 restent inchangés.

## S468 — validation indépendante de l'absence de perte d'attaques sur les données IDMT annotées

[GitHub Actions S468 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38552723494).

Le scorer ouvre les 47 véritables attaques XML seulement après contrôle SHA du fichier S467, et réemploie les identifiants / K d'origine S453/S462. **Résultat :**

| Métrique rétrospective IDMT | Observé |
|---|---:|
| Attaques annotées conservées | **47/47** |
| Vraies attaques MIDI 40–88 conservées | **45/45** |
| Vraies attaques polyphoniques conservées | **21/21** |
| Vraies attaques que S80 avait classées K0 conservées | **7/7** |
| Candidats éloignés des attaques annotées proposés à écarter | **33/418** |
| Dont prédits K0 par S80 | **33/417** |
| K d'origine ou notes réellement modifiés | **0** |

Sur ces cinq enregistrements, les scores moyens des quatre modèles différencient fortement les événements appariés aux attaques réelles et les candidats très éloignés. **Indice de séparabilité** AUC diagnostique ~0,995, *NON* AUC de classification des faux événements : les 418 non appariés n'ont pas de vérité binaire « fausse attaque ». Ne pas publier cet AUC comme précision réelle sur l'ensemble du monde.

## Verdict et prochaines décisions

**Meilleur candidat expérimental à conserver : S466 → S467 (détecteur binaire isolé + garde des K non nuls + consensus 4/4).** Sa validation IDMT S468 ne détecte aucune perte des 47 vraies attaques malgré l'écartement proposé de 33 candidats. Le filtre simple S465 échoue à garantir cette propriété.

**Pas de promotion.** Huit vrais onsets GuitarSet restent perdus dans l'évaluation composition-held ; IDMT a été exposé historiquement en S299 et ce test n'est pas un nouveau jeu aveugle. L'activation exigerait un nouveau corpus annoté après gel du modèle (plusieurs timbres/instruments et niveaux sonores), et des résultats de rappel / faux événements définis par une véritable annotation négative ou une définition d'événements non ambigüe.

**Exact-K et noms de notes n'ont pas été améliorés** par S465–S468 : S80 conserve ses scores historiques et 7/21 Exact-K sur le sous-ensemble poly IDMT, S446 demeure un correcteur des noms de notes sous K fixé. L'étape suivante n'est pas de fabriquer un +XX % Exact-K, mais de spécialiser la seconde tête sur des erreurs K2/K3/K4 avec des preuves indépendantes des seules amplitudes, tout en gardant le nouveau détecteur de naissance comme signal auxiliaire (et non filtre destructif).
