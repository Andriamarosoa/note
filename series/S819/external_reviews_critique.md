# S819 — Contre-audit scientifique des trois vrais avis NVIDIA (non consensuel par défaut)

## Preuve des consultations réellement effectuées

[GitHub Actions #39552451395](https://github.com/Andriamarosoa/note/actions/runs/39552451395) **SUCCESS** : 3 consultations distinctes réalisées auprès de **google/gemma-4-31b-it**, **nvidia/nemotron-3.5-lightning-30b-a3b**, **z-ai/glm-5.3-flash** et six audits GitHub techniques distincts (Données, Acoustique, Architecture, Apprentissage, Métriques, LIVE). Artefact des 3 réponses brutes et 6 contrôles [S819_THREE_REAL_NVIDIA_SIX_LOCAL_CONSENSUS](https://github.com/Andriamarosoa/note/actions/runs/39552451395). Chaque modèle a reçu le **même** `series/S819/model_brief.md` et le même prompt système, empreintes SHA vérifiées par le script fail-closed. Consultation par `NVIDIA_API_KEY` GitHub Actions, non exposé. **Aucun des six agents personnalisés ChatGPT Work n'a été invoqué : les six rôles du run sont des scripts de vérification déterministes.**

L'obtention de 3 avis n'est pas une validation par 3 experts humains ni une amélioration mesurée : la plausibilité de chaque réponse doit être falsifiée sur le code et les résultats du dépôt.

## Hypothèses indépendantes comparées sans leur donner les avis des autres

| Hypothèse | Gemma | Nemotron | GLM | Verdict contradictoire de l'assistant |
|---|---|---|---|---|
| Confusions fondamentales/harmoniques K4–K6 | Oui | Indirecte | Oui | **Non prouvée** par le seul taux d'erreur. Requiert contrôle audio fondamentale vs partiels, comparaison audio sans labels de test pour sélection |
| Échantillonnage cap=130 et poids inverses S818 | Oui | Indirecte | Oui | **Expérience d'ablation contrôlée possible**, mais le code restitue explicitement le prior naturel avec `len(at)/n_sampled`. Pas de preuve que les poids eux-mêmes sont faux |
| Arbitrage des têtes S817 / sélecteurs S815–S818 | Oui | Oui | Oui | **Priorité au contre-test** : S817 0/306536 décisions changées ; quatre compositions ont choisi `base`, l'une `max_high` avec seuil .90, ce qui peut neutraliser les nouveaux experts sans mesurer leur discrimination |
| Transfert vers Rock3 / SS1 | Oui | Oui | Oui | Scores Rock3 51,87 % et SS1 51,15 % sont réels pour S818 ; cause timbre/domain shift **non identifiée** par données agrégées |
| Labels et temporalité MIDI256 | Oui | Oui | Oui | **À auditer sans modifier la résolution**. Le pas de 256 samples = 5,805ms, pas durée garantie d'une fenêtre DSP ni latence physique. N'autorise ni future frame ni hold |
| Nouvelle architecture/latence | Hypothèses | Hypothèses | Hypothèses | **Aucune performance prédite n'est une mesure** ; veto S708 et six axes conservé |

## Assertions de modèles explicitement rejetées

- **Nemotron H4** qualifie l'absence de lookahead/hold de « fuite temporelle » : faux, la causalité stricte **empêche** une fuite du futur. Il propose aussi « 128 ms » alors que 128 échantillons à 44,1 kHz font **2,90 ms** ; changer les hops violerait le contrat MIDI256 demandé. Sa suggestion d'ajuster les seuils sur « 48 pistes de calibration » confond les 48 pistes d'entraînement et les **16 pistes de calibration**.
- **GLM H2** écrit que sur S818 « under ET over augmentent tous deux » par rapport à S815 : arithmétiquement faux. S815 under=65530, over=41935 ; S818 under=73512 **augmente**, over=40205 **diminue**. Les poids de S818 reconstruisent les proportions originales par classe (vérification dans `training_samples`) ; leur variance ou la répartition des exemples peuvent rester problématiques mais une hypothèse de « prior détruit » n'est pas démontrée.
- **Gemma** décrit la décroissance K1→K6 comme « linéaire » alors que K2 42,39 % → K3 46,19 % **augmente**. Il assimile par endroits le pas MIDI256 à une **fenêtre spectrale**, sans démonstration du support temporel DSP ; et réclame d'« unlock CAL32/HELD/TEST » pour comprendre la généralisation : **interdit** tant que les politiques ne sont pas gelées et les splits réservés protégés.
- Aucun des trois ne peut prouver une confusion harmonique/octave spécifique à Rock3 ou SS1 à partir des seules données agrégées ; aucune écoute ou analyse des WAV et annotations individuels n'a été effectuée par les conseillers. Les pistes sont des hypothèses, pas des résultats.

## Décision sur la prochaine expérience S819

**Mesurer d'abord la neutralisation des nouvelles têtes S817** sans les réentraîner : 5 modèles gelés, 80 WAV PCM authentiques rechargés, comparaison de politiques pré-déclarées `base`, `override`, `max_high` aux seuils .50/.70/.90, sur chaque composition laissée dehors, avec distinction explicite CAL interne et outer test (diagnostic a posteriori, jamais choisir sur outer). Reproduire d'abord **H0 142213/249411** et **S815/S817 141946/249411**. Mesurer activation réelle de la porte K>=4, combien d'overrides potentiels, sous/surcomptages par vrai K, et tests de causalité. Ce diagnostic discrimine :
1) aucun potentiel de changer les décisions même si porte active ;
2) des changements existent mais le sélecteur calibration ne les choisit pas ;
3) le sélecteur peut changer les décisions mais les erreurs augmentent en distribution tierce.

Ensuite seulement une ablation strictement factorielle des poids et du cap de S818, si les diagnostics la justifient ; ne pas modifier simultanément famille de modèle et règle de score. **Aucune promotion S819, aucune intégration MIDI LIVE, aucune unification** avant validation externe et six métriques indépendantes >85 %. Le résultat FIT80 ne peut pas être présenté comme une preuve de généralisation après autant d'itérations.
