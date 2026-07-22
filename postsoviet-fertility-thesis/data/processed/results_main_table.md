## Panel A — Regional fertility difference (Central Asia vs rest)

| | M1: raw gap | M2: + controls | M2h: Mundlak |
|---|---|---|---|
| | Pooled, year FE | Pooled, year FE | Between/within |
| Central Asia dummy | +1.314<br>[+0.994, +1.635] | +0.971<br>[+0.740, +1.201] | +0.964<br>[+0.689, +1.238] |
| *Bootstrap p (CA, script 19-G)* | p = 0.001 | p = 0.006 | p = 0.033 |
| log GDP per capita PPP (lag) | — | +0.009<br>[-0.244, +0.263] | +0.182(w)<br>[-0.362, +0.727] |
| Urban population %  (lag) | — | -0.012<br>[-0.026, +0.002] | +0.006(w)<br>[-0.043, +0.056] |
| Remittances % GDP  (lag) | — | +0.001<br>[-0.011, +0.013] | +0.001(w)<br>[-0.014, +0.015] |
| Under-5 mortality  (lag) | — | +0.006<br>[-0.003, +0.014] | -0.004(w)<br>[-0.021, +0.013] |
| Year FE | Yes | Yes | Yes |
| Country effects | No | No | means (between) |
| Observations | 315 | 315 | 315 |
| R² | 0.842 | 0.908 | 0.918 |
| R² type | R²(overall) | R²(overall) | R²(overall) |

## Panel B — Within-country associations

| | M4: two-way FE | FD | FD + year FE |
|---|---|---|---|
| | Country+year FE | Within (Δ) | Within (Δ)+yr |
| log GDP per capita PPP (lag) | +0.183<br>[-0.328, +0.693] | +0.125<br>[-0.014, +0.265] | +0.054<br>[-0.210, +0.318] |
| Urban population %  (lag) | +0.006<br>[-0.043, +0.055] | -0.004<br>[-0.030, +0.022] | +0.025<br>[-0.007, +0.058] |
| Remittances % GDP  (lag) | +0.000<br>[-0.013, +0.014] | +0.000<br>[-0.004, +0.005] | +0.001<br>[-0.004, +0.005] |
| Under-5 mortality  (lag) | -0.004<br>[-0.021, +0.013] | -0.008<br>[-0.015, -0.002] | -0.000<br>[-0.007, +0.007] |
| Year FE | Yes | No | Yes |
| Country effects | estimated (FE) | differenced out | differenced out |
| Observations | 315 | 301 | 301 |
| R² | 0.296 | 0.056 | 0.257 |
| R² type | R²(within) | R²(differenced) | R²(differenced) |

*Brackets are 95% cluster-robust confidence intervals, not raw standard errors; significance stars are not used (see script docstring).*  
*All panels estimated on the controls-complete sample (N=315). Panel A: 'ca' is the between-country Central Asia premium in M1/M2/M2h; M2h's control rows are marked (w) as within-country deviations (not the same estimand as 'ca' or as M1/M2's pooled controls); M2h's collinear between-country control coefficients are not tabulated — see diagnostics. Panel B: within-country associations only, no between-country gap estimated. M4: 'ca' absorbed by country FE; FD / FD+yr: 'ca' differenced away. FD / FD+yr R² is computed on first-differenced data, not the FE within-R². Cluster count = 14; wild-cluster bootstrap (script 19, section G) was run for the CA coefficients in M1, M2, M2h (Panel A's 'Bootstrap p (CA)' row) and for CA x remittances / CA x urbanisation. The asymptotic clustered p-value for M2h 'ca' is < 0.01, but its wild-cluster bootstrap p-value is 0.033 (significant at 5%, not 1%) — the CI and bootstrap-p row above reflect that gap rather than the asymptotic p-value alone. Under-5 mortality is significant in FD without year FE but not once year effects are added, i.e. it is sensitive to the inclusion of common year effects.*
