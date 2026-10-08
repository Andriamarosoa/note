# Série 1 — ouvrir les réponses K0/K1 existantes

Poids inchangés ; 14 politiques annoncées et conservées. Les deux témoins reproduisent exactement les décisions archivées.
Ces chiffres sont du développement exposé, sans validation indépendante.

| Politique | Global | Poly | Net global / freeze | Net poly / freeze |
|---|---:|---:|---:|---:|
| S8__low_gate_0.50 | 82.3973% | 33.8253% | +415 | -32 |
| S8__low_gate_0.65 | 82.2489% | 35.3961% | +327 | +84 |
| S8__low_gate_0.80 | 82.1579% | 36.2085% | +273 | +144 |
| S8__low_gate_0.90 | 82.0853% | 36.3981% | +230 | +158 |
| S8__low_gate_0.95 | 82.0432% | 36.4252% | +205 | +160 |
| S8__open_argmax | 82.2995% | 29.6953% | +357 | -337 |
| S8__restricted_control | 81.7987% | 35.0711% | +60 | +60 |
| coherent__low_gate_0.50 | 82.5709% | 33.6764% | +518 | -43 |
| coherent__low_gate_0.65 | 82.3231% | 35.3690% | +371 | +82 |
| coherent__low_gate_0.80 | 82.1697% | 36.1950% | +280 | +143 |
| coherent__low_gate_0.90 | 82.0803% | 36.4252% | +227 | +160 |
| coherent__low_gate_0.95 | 82.0263% | 36.4523% | +195 | +162 |
| coherent__open_argmax | 82.4630% | 28.9641% | +454 | -391 |
| coherent__restricted_control | 81.7498% | 34.6784% | +31 | +31 |

Le précédent meilleur candidat est à 81,9724 % global et 36,4658 % poly.
La garde `coherent__low_gate_0.95` atteint 82,0263 % / 36,4523 % : 33 corrections nouvelles (27 vrais K0, 6 vrais K1), 1 régression vraie K2, net +32 face à ce candidat.
L’ouverture libre du critique monte à 82,4630 % global mais chute à 28,9641 % poly. Elle ne satisfait donc pas le critère global et poly simultanément.

La régression de la garde stricte, indice global 30476, prédit K0 avec une probabilité de 0,9856112 alors que la vérité est K2. Cette confiance élevée est erronée. Ce cas ne suffit pas à établir une nouvelle catégorie ou une cause acoustique.

L’union oracle des 255 propositions, des 211 anciennes politiques et des 14 nouvelles politiques couvre désormais 51 461 bonnes réponses, soit 86,7676 % global et 61,2729 % poly. Elle ajoute 1 025 erreurs corrigibles, dont 1 012 K0/K1 et 13 polyphoniques. Ce maximum utilise la vérité pour choisir ; aucun modèle n’atteint ici ce score.

Les 14 politiques et leurs probabilités sources restent disponibles. L’extraction acoustique complète et les nouveaux producteurs K0–K6 constituent la série 2 annoncée avant ces résultats.
