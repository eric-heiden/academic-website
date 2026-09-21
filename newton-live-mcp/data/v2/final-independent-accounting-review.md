# Final independent accounting review

Recorded 2026-09-21T02:51:54.192784+00:00. Frozen simulation source `fe4fda0199adae250d0ed1d5e37cb91f98c89f77`.

**0 issues found.** All 61 completed rows, 27 cost groups, 39 paired comparison blocks, 117 metric-statistics blocks, and 27 real-data quality-figure rows reconcile. All 54 primary rows exactly match the preserved earlier independent review. All 61 contexts are eligible; 58 pass physical quality. Each primary method passes 9/9 existing cases and 8/9 real-data repeats. Corrected IPython passes all seven sensitivity cases.

The separate recount reads original JSON and JSONL evidence and registered metadata. It imports no report-builder or Newton code. The raw event, candidate, process, configuration, and scalar-quality checks were rerun for all 61 contexts. It does not resimulate physics or rederive trajectory errors from saved arrays. The original primary review and all source/trial files remain unchanged.

## Primary paired results

Ratios below one favor the first named method. Raw tokens are input plus output; uncached tokens are input minus cached input plus output. Ratios use jointly successful eligible same-case pairs. The unchanged all-cost totals, including failures, remain in the primary review and final comparison.

| Cohort | Numerator / denominator | Pairs | Time | Raw tokens | Uncached tokens |
|---|---|---:|---:|---:|---:|
| Existing | Newton MCP / Edit/restart | 9 | 0.7968034261294712 | 0.6859520915041022 | 0.8169673748071427 |
| Existing | Newton MCP / Upstream IPython | 9 | 1.1057634426549907 | 1.0779489450116173 | 0.8296598704010554 |
| Existing | Edit/restart / Upstream IPython | 9 | 1.3877493574874729 | 1.5714638942903068 | 1.0155361107252405 |
| Real data | Newton MCP / Edit/restart | 7 | 1.0453397984032953 | 1.1501685533609745 | 1.2710593922164861 |
| Real data | Newton MCP / Upstream IPython | 7 | 1.2252185680154786 | 1.479781919783498 | 1.2185948574905472 |
| Real data | Edit/restart / Upstream IPython | 8 | 1.1935957138011746 | 1.2752358818170755 | 0.9442652228319035 |

All three physical failures remain exactly as previously recorded: edit/restart repeat 3, recording 25, joint 5 normalized torque RMSE 0.5052716849339283; upstream IPython repeat 3, recording 30, joint 5 value 0.5005334121902201; Newton repeat 5, recording 30, joint 5 value 0.5022877620994345. The unchanged limit is 0.5. Matching logged training, fresh training, complete finite coverage, and every other held-out gate pass. Newton real-data pairs exclude repeats 3 and 5; edit/restart versus IPython excludes repeat 3.

## Corrected-IPython sensitivity

All seven contexts pass. Their combined startup-inclusive time is 1026.0849221898243 seconds; input plus output is 2,056,729 tokens; uncached input plus output is 336,409 tokens; cached input is 1,720,320 tokens. There are 57 completed candidates, including 16 failed candidates, seven trial simulation processes, and ten independent verifier processes. No final usage, completed context, or failed candidate is omitted.

| Subset | Corrected IPython / comparator | Pairs | Time | Raw tokens | Uncached tokens |
|---|---|---:|---:|---:|---:|
| Seven matched cases | Corrected IPython / Newton MCP | 7 | 0.8028969900147164 | 0.7531039838198486 | 1.0326517653179503 |
| Seven matched cases | Corrected IPython / Edit/restart | 7 | 0.7861633357673871 | 0.6791198992160398 | 1.054731383247144 |
| Seven matched cases | Corrected IPython / Upstream IPython | 7 | 0.8806794244416661 | 0.8186142915028413 | 0.8718807902197743 |
| Real-data repeats 0–2 | Corrected IPython / Newton MCP | 3 | 0.7273776602954989 | 0.6057967304309295 | 1.1487131904878376 |
| Real-data repeats 0–2 | Corrected IPython / Edit/restart | 3 | 0.7680750554371379 | 0.6828467965335855 | 1.4530515663081351 |
| Real-data repeats 0–2 | Corrected IPython / Upstream IPython | 3 | 0.7966277437835972 | 0.7174640685317925 | 1.1741584460534926 |

| Sensitivity context | Time (s) | Raw tokens | Uncached tokens | Candidates | Largest measured gate / limit |
|---|---:|---:|---:|---:|---:|
| panda_real-ipython_fixed-1 | 274.13541658455506 | 478,123 | 77,355 | 17 | 0.899251845 |
| hug-ipython_fixed-0 | 64.39126262674108 | 149,254 | 23,686 | 1 | 0.386522216 |
| panda-ipython_fixed-0 | 50.855467951856554 | 135,997 | 15,037 | 1 | 0.133060664 |
| panda_real-ipython_fixed-2 | 213.604408999905 | 457,861 | 71,941 | 12 | 0.909236917 |
| panda_calibration-ipython_fixed-0 | 78.6056092469953 | 160,161 | 19,873 | 9 | 0.575692038 |
| panda_real-ipython_fixed-0 | 289.2207480738871 | 541,292 | 113,004 | 16 | 0.936704003 |
| allegro-ipython_fixed-0 | 55.27200870588422 | 134,041 | 15,513 | 1 | 0.164769513 |

The last column is only a compact coverage check: the maximum of each recorded physical metric divided by its own threshold, considering both pooled and per-recording values. It is not a new quality objective or an aggregate score. All values are below one. Full per-gate values, limits, sample counts, and reference identities are preserved in the JSON review.

## Statistics, exclusions, and interval orientation

The audit independently enumerates every sign pattern for the two-sided paired log-ratio permutation test, recounts the exact two-sided sign test, and regenerates 20,000 bootstrap resamples of whole matched pairs with seed 20260920. It reproduces linear-interpolated 2.5th and 97.5th percentiles. All 117 metric-statistics blocks agree within floating-point tolerance. Included and excluded case lists agree exactly.

Primary tables display the reciprocal of the builder’s stored ratio. Their confidence limits correctly transform from [lower, upper] to [1/upper, 1/lower]. Sensitivity tables retain corrected-IPython/comparator orientation. The performance figure uses the same audited pair exclusions and reciprocal Newton/comparator ratios; all real-data quality rows include both pooled and per-recording gates and retain the three failures.

Percentile-bootstrap intervals and exact tests can disagree about a threshold crossing in these small samples. For corrected versus upstream IPython across seven cases, the time ratio is 0.8806794244416661 with interval [0.6904559696593553, 1.0554019045049505]; the raw-token ratio is 0.8186142915028413 with interval [0.658037816696518, 0.9851249432346949]. Their exact paired permutation p-values are 0.359375 and 0.171875. The real-data sensitivity has only three pairs; a two-sided exact sign test cannot yield a p-value below 0.25 even if all three move in the same direction. These are unadjusted descriptive comparisons.

## Faithful interpretation

Newton MCP reduced paired time and raw-token use versus edit/restart on the existing cases. The real measured identification task did not reproduce that paired advantage: Newton’s successful-pair geometric time and raw-token ratios exceed one against both edit/restart and upstream IPython. All primary methods have the same physical success counts, with only narrow per-recording torque-threshold misses. The data therefore do not support a general claim that Newton MCP is faster or more token-efficient than the actual upstream IPython MCP or edit/restart for full-arm identification.

The all-cost total time for Newton is lower than edit/restart on real-data trials, while its successful-pair geometric time ratio is above one. These are different estimands: all-cost totals include every trial and weight long trials more heavily; paired geometric ratios exclude unmatched failures and weight case ratios equally. Both must remain visible.

Corrected IPython has lower descriptive time and raw-token ratios on the seven selected matched cases, but this is a later execution block, not an isolated randomized test of its two code changes. Agent search paths, candidate counts, and cached-token composition vary. Three repeats on the same recorded identification problem do not establish performance across robots or datasets; neither do nine repeats on that same problem. No causal speedup or population-wide superiority follows.

## Evidence commitments

- Registration: `2a9af47f10b2dcc752e060db846e27a48aad203bee07ecc8290a676305108f22`
- Final comparison: `e4e6a04393c8c76bb8f3a22a83f24418243dbdd667bd5a957fb6387c1564e757`
- Preserved primary review: `c50a8e364483a56969f43145725f843c2210e9aa679471530a5578c15a1971f9`
- Figure data: `cfeb00a4d5ccfab4c803694f0e0b66d7d7c21ff6876c576597ee7fbe6183f1f8`
- Recount script: `94e2f96461d0bdf714486fc1c3384bf8372fc9d14d56680e6bec5e98a5bb574c`

Full precision and all statistical blocks: `final-independent-accounting-review.json`. Recount source: `final-independent-accounting-review.py`; execution output: `final-independent-accounting-review.log`.
