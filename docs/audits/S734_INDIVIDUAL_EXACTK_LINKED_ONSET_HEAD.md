# S734 — Une sixième tête Onset individuelle reliée à Exact-K

**18 octobre 2026 — R&D, zéro promotion Live.**

## Demande et implémentation

La banque S733 comportait cinq propositions d'Onset, toutes fondées sur des
prédictions acoustiques déjà disponibles. Une sixième tête **H5** a été ajoutée :
`OneTargetClassConditionalSelector.k_linked_onset`.

H5 est un **réseau neuronal indépendant pour chaque hauteur MIDI 40–88 et
chaque décision causale**, qui reçoit :

- les 16 indices acoustiques/rythmiques locaux Onset, Sustain, Release,
  nouveauté harmonique, voisinage, source déjà entendue et progression ;
- la distribution **complète K0..K6 de S708, gelée** (et non le vrai K) ;
- le **nombre de notes MIDI déjà actives** émis par le flux S708+S707 à l'instant,
  plus le `K-room` estimé par la scène auditive S731 ;
- la probabilité Onset originale S708, pour prendre une décision relative.

H5 émet un seul nombre, **P(attaque de cette hauteur)**. Elle n'émet jamais
une prédiction K0..K6 et **ne modifie pas le modèle Exact-K lui-même**.
L'attention entre têtes et son réseau d'arbitrage
KEEP/ADOPT/SUPPRESS de S733 prennent ensuite six propositions au lieu de cinq.

**Important : le lien au K est SOUPLE.** Si S708 sous-estime Exact-K,
H5 reste admissible même quand K0 est majoritaire. Un blocage strict
`active_count >= argmax(K)` empêcherait justement la récupération de notes
réelles dans les accords.

## Mesures, limites et anti-fuites

La tête H5 et le sélecteur Onset sont entraînés uniquement sur FIT80 pour la
même fonction Onset. Sustain, Release, K et S707 restent figés.
Le modèle H5 reçoit une perte supervisée d'attaque vraie ou absente
(BCE), **sans aucune supervision à reproduire le K réel en inférence**.

Les 5 têtes historiques conservent leurs profils de risque correction/régression
OOF inter-compositions. **La nouvelle H5 n'en a pas** : son profil historique
est nul et ses risques auxiliaires sont masqués de la perte, plutôt que de
fabriquer de fausses observations hors-fold. Pour H5, la validation de
généralisation ne vient que des événements reconstruits sur CAL, puis des
contrôles HELD/DEV (historiquement exposés).

Le code corrige aussi l'échec d'intégration de S733 : la variable `original`
portant le rapport S708 avait été écrasée par des événements MIDI FIT ;
`baseline_events` empêche désormais ce masquage.

L'expérience n'est pas une validation de supériorité H5 **tant qu'une
ablation H0..H4 / H0..H5 à données et budgets identiques n'est pas analysée**.
Ne jamais déduire un score élevé d'une amélioration du posterior seul ;
il faut de vrais NoteOn, des faux positifs, Exact-K actif et global,
Offset et notes complètes, y compris les cas K sous-estimés.

CI: `.github/workflows/note-s734-klinked-onset-fit.yml`
branchement H5: `src/causal_note/s733_one_target_selector.py`
apprentissage: `scripts/s733_train_onset_only.py`
contrats: `test/test_s733_one_target_selector.py`

**Aucun checkpoint S734 n'est promu, aucune modification du Live, aucun TEST ouvert.**
