# LIAR2 Credibility Leakage. Detection and Leak-Free Correction (PSA / WSA / LFA)

## Original: Authors' Reported Values

These are the credibility-feature results as originally reported by the LIAR2 authors (Xu and Kechadi, [*An Enhanced Fake News Detection System With Fuzzy Deep Learning*](https://doi.org/10.1109/ACCESS.2024.3418340)) in their [`chengxuphd/liar2`](https://github.com/chengxuphd/liar2) repository, included here as the published reference point. The PSA tables provide our controlled re-run of the same pre-split-aggregation (leaky) construction for direct comparison.

All values are as reported by the original authors (a single published run); the PSA, WSA, and LFA tables instead report the mean over five random seeds (7, 13, 42, 77, 123).

## Segmentation results

| **Train**              | **Test & Val.**       | **Val. Accuracy** | **Val. F1-Macro** | **Val. F1-Micro** | **Test Accuracy** | **Test F1-Macro** | **Test F1-Micro** | **Mean** |
|------------------------|------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|----------|
| LIAR (Original .8)     | LIAR (Original .2)     | 0.4673            | 0.4490            | 0.4577            | 0.4649            | 0.4701            | 0.4649            | 0.4623   |
| LIAR (.8)              | LIAR (.2)              | 0.6140            | 0.5888            | 0.5833            | 0.6084            | 0.6138            | 0.6084            | 0.6028   |
| NEW (.8)               | NEW (.2)               | 0.7676            | 0.6706            | 0.7473            | 0.7592            | 0.6952            | 0.7592            | 0.7332   |
| LIAR (1.)              | NEW (1.)               | 0.7361            | 0.6963            | 0.7301            | 0.7361            | 0.7057            | 0.7361            | 0.7234   |
| NEW (1.)               | LIAR (1.)              | 0.5575            | 0.5203            | 0.5199            | 0.5575            | 0.5626            | 0.5575            | 0.5459   |
| LIAR (1.) + NEW (.557) | NEW (.443)             | 0.7867            | 0.7249            | 0.7696            | 0.7827            | 0.7389            | 0.7827            | 0.7643   |
| LIAR (.635) + NEW (1.) | LIAR (.365)            | 0.6308            | 0.6118            | 0.6073            | 0.6211            | 0.6270            | 0.6211            | 0.6199   |
| LIAR (.8) + NEW (.8)   | LIAR (.2) + NEW (.2)   | 0.6977            | 0.6671            | 0.6771            | 0.6892            | 0.6828            | 0.6892            | 0.6839   |
| LIAR + NEW (Mix .8)    | LIAR + NEW (Mix .2)    | 0.6974            | 0.6570            | 0.6676            | 0.7021            | 0.6961            | 0.7021            | 0.6871   |

## Ablation results

| **Feature**              | **Val. Accuracy** | **Val. F1-Macro** | **Val. F1-Micro** | **Test Accuracy** | **Test F1-Macro** | **Test F1-Micro** | **Mean** |
|--------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|----------|
| Statement                | 0.3174       | 0.1957       | 0.3117       | 0.3197       | 0.2380       | 0.3197       | 0.2837       |
| Date                     | 0.2912       | 0.1879       | 0.2912       | 0.3079       | 0.1775       | 0.3079       | 0.2606       |
| Subject                  | 0.3243       | 0.2311       | 0.3183       | 0.3267       | 0.2271       | 0.3267       | 0.2924       |
| Speaker                  | 0.3283       | 0.2250       | 0.3174       | 0.3310       | 0.2462       | 0.3310       | 0.2965       |
| Speaker Description      | 0.3322       | 0.2444       | 0.3250       | 0.3280       | 0.2444       | 0.3280       | 0.3003       |
| State Info               | 0.2930       | 0.1577       | 0.2950       | 0.2979       | 0.1521       | 0.2979       | 0.2489       |
| Credibility History      | 0.5007       | 0.4696       | 0.4985       | 0.5057       | 0.4656       | 0.5057       | 0.4910       |
| Context                  | 0.2982       | 0.1817       | 0.2982       | 0.3132       | 0.1791       | 0.3132       | 0.2639       |
| **Justification**        | **0.5964**   | **0.5657**   | **0.5827**   | **0.6115**   | **0.5968**   | **0.6115**   | **0.5941**   |
| All without
| **Statement**            | **0.7079**   | **0.6734**   | **0.6822**   | **0.7182**   | **0.7108**   | **0.7182**   | **0.7018**   |
| Date                     | 0.6931       | 0.6572       | 0.6680       | 0.7078       | 0.6993       | 0.7078       | 0.6889       |
| Subject                  | 0.7000       | 0.6579       | 0.6681       | 0.7078       | 0.7013       | 0.7078       | 0.6905       |
| Speaker                  | 0.6944       | 0.6648       | 0.6757       | 0.7043       | 0.6942       | 0.7043       | 0.6896       |
| Speaker Description      | 0.6892       | 0.6640       | 0.6739       | 0.7169       | 0.7073       | 0.7169       | 0.6947       |
| State Info               | 0.7074       | 0.6625       | 0.6729       | 0.7099       | 0.7016       | 0.7099       | 0.6940       |
| Credibility History      | 0.6025       | 0.5717       | 0.5900       | 0.6185       | 0.6046       | 0.6185       | 0.6010       |
| Context                  | 0.7005       | 0.6622       | 0.6720       | 0.7043       | 0.6967       | 0.7043       | 0.6900       |
| Justification            | 0.5285       | 0.4898       | 0.5153       | 0.5340       | 0.5148       | 0.5340       | 0.5194       |
| Statement +
| Date                     | 0.3431       | 0.2540       | 0.3343       | 0.3380       | 0.2514       | 0.3380       | 0.3098       |
| Subject                  | 0.3548       | 0.2759       | 0.3513       | 0.3375       | 0.2580       | 0.3375       | 0.3192       |
| Speaker                  | 0.3618       | 0.2862       | 0.3539       | 0.3476       | 0.2640       | 0.3476       | 0.3269       |
| Speaker Description      | 0.3583       | 0.2814       | 0.3531       | 0.3667       | 0.2886       | 0.3667       | 0.3358       |
| State Info               | 0.3317       | 0.2367       | 0.3294       | 0.3328       | 0.2362       | 0.3328       | 0.2999       |
| Credibility History      | 0.5067       | 0.4737       | 0.5084       | 0.5244       | 0.5000       | 0.5244       | 0.5063       |
| Context                  | 0.3361       | 0.2682       | 0.3391       | 0.3458       | 0.2560       | 0.3458       | 0.3152       |
| Justification            | 0.6017       | 0.5578       | 0.5796       | 0.6176       | 0.6026       | 0.6176       | 0.5962       |
| **All**                  | **0.6974**   | **0.6570**   | **0.6676**   | **0.7021**   | **0.6961**   | **0.7021**   | **0.6871**   |
