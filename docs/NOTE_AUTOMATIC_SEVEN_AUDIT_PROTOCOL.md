# NOTE — automatisation du septième conseiller avec les six audits

## Modèle d'exécution vérifié en S802

La [série S802](https://github.com/Andriamarosoa/note/actions/runs/39359182007) a effectivement exécuté dans un **seul run** les six rôles déterministes NOTE (data, acoustics, architecture, training, metrics, live) et le conseiller indépendant NVIDIA `z-ai/glm-5.3-flash`, **en parallèle après la préparation des données FIT**.

La configuration est **déclenchée par push** dans la branche scientifique : le bouton `workflow_dispatch` est optionnel, pas obligatoire. Aucune intervention manuelle ou saisie de clé à chaque boucle. La clé est le secret GitHub Actions `NVIDIA_API_KEY` et ne doit jamais être révélée ou copiée dans des scripts.

## Pour toute nouvelle série S803, S804, etc.

Le workflow scientifique de chaque nouvelle série **doit** inclure la structure logique suivante (illustration ; remplacer `S803` et les noms de jobs par ceux de la nouvelle série) :

```yaml
  six_audits:
    needs: [smoke, experiment]
    strategy:
      matrix:
        role: [data, acoustics, architecture, training, metrics, live]
    # ... vrais rapports d'audit par rôle ...

  nvidia_seventh_adviser:
    needs: [smoke, experiment]  # identique à six_audits -> PARALLÈLE
    uses: ./.github/workflows/note-reusable-seventh-adviser.yml
    with:
      series: S803
    secrets: inherit

  consensus:
    if: always()
    needs: [smoke, experiment, six_audits, nvidia_seventh_adviser]
    # ... télécharger les six rôles ET S803_SEVENTH_NVIDIA_INDEPENDENT_REVIEW ...
    # python -m scripts.s787_audit_consensus --reports-dir ... --out ...
    # python -m scripts.note_seventh_consensus --series S803 \
    #   --six-consensus ... --seventh-advice ... --out ...
```

La requête NVIDIA n'attend **pas** la fin des six audits, ce qui permet leur parallélisation. **Le consensus final**, en revanche, attend sept résultats : six rapports déterministes PASS et un avis externe réellement reçu et non vide. Tout échec / saut / indisponibilité du fournisseur bloque le 7/7 et **n'autorise aucune promotion**.

Le client `scripts/note_seventh_adviser.py` n'expédie que les trois derniers `resultat.md` scientifiques, avec blocage des clés détectables ; ni audio, ni archives de prédictions OOF brutes, ni code source. Le conseiller donne des propositions de recherche mais ne modifie pas le code, les architectures, les scores CAL32 ou la chaîne MIDI LIVE.

## Éviter les boucles indésirables

Les mises à jour documentaires post-run de `resume.md` et `resultat.md` ne doivent pas déclencher une nouvelle inférence NVIDIA ; configurer les filtres `paths` pour les scripts scientifiques, les tests et les workflows concernés. Sur une nouvelle série, chaque nouveau **run scientifique complet** déclenche une seule consultation du septième conseiller. Les anciens workflows NVIDIA séparés sont à proscrire (doublons et appels inutiles).

## Distinction importante

Les six auditeurs GitHub sont six scripts/rôles déterministes ; ils ne sont **pas** les six agents ChatGPT Work personnalisés. NVIDIA est le septième relecteur scientifique LLM externe. Le succès 7/7 valide le **protocole d'audit**, pas l'atteinte des six métriques musicales à 80 %.


## Extension depuis S804 : trois modèles NVIDIA, neuf preuves indépendantes

Le protocole **7/7** ci-dessus reste l'historique valide S802/S803 ; il ne constitue pas la nouvelle norme de revue code/données. S804 introduit un protocole **9/9** : les six audits déterministes habituels ET trois modèles NVIDIA indépendants sur les preuves GitHub signées des 80 pistes FIT. Voir `.github/workflows/note-s804-deep-nine-audits.yml`.

- Audit externe d'architecture : Gemma 4 31B ;
- Audit externe des données : Nemotron 3.5 Lightning ;
- Contre-audit externe : Poolside Laguna XS 2.1.

Les trois jobs externes et six locaux s'exécutent **en parallèle seulement après** production d'un bundle de preuve : scan AST de tous les Python suivis par Git, extraits numérotés des chemins critiques, validation des 80 cartes OOF, statistiques réelles des erreurs par pitch/composition et contrôles SHA256. Le modèle ne lit pas chaque ligne du dépôt ni les fichiers audio entiers : cette limite doit être explicitement déclarée. CAL32, HELD et TEST sont interdits. La réussite de la consultation d'une API ne prouve ni absence de bug, ni F1 musical, ni latence physique.

Pour chaque **future série S805+**, faire partir les trois demandes du même bundle et bloquer le consensus si un modèle ne répond pas ou si les preuves d'audit sont périmées, non authentifiées ou sur un autre SHA. Garder les anciens rapports S802/S803 inchangés. N'activer jamais de workflows historiques NVIDIA en parallèle du nouveau mécanisme pour éviter un double déclenchement. Les nouvelles série doivent explicitement adopter ce flux ; l'ajout documentaire ne modifie pas les workflows déjà existants.
