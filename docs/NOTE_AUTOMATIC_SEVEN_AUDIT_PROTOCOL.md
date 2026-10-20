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
