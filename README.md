# gta6-trophy-rarity-forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jpmax2027/gta6-trophy-rarity-forecaster/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jpmax2027/gta6-trophy-rarity-forecaster/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts the earn rate of every Grand Theft Auto VI trophy on Exophase (PlayStation 5) at 2026-11-26 00:00 UTC — the percent of the players Exophase tracks who will have earned it one week after launch — as five quantiles per trophy (5/25/50/75/95%), scored by mean pinball loss on the logit scale.

## Approach

The program (`main.py`, Python 3.13, standard library only) reads the case's recorded reads of the trophy page and models each trophy on the logit scale, the scale it is scored on:

- A recency-weighted least-squares line of logit(rate) against ln(1 + hours since release), read at the target. Log-time makes the extrapolation saturate the way trophy curves do, and each trophy's slope is shrunk toward its tier's pooled slope by the fit's effective sample size.
- A second forecast from the counts behind the rate: the same line fitted to the trophy's earner count (rate × players tracked) and to the players-tracked count itself, divided at the target. This sees the dilution a growing tracked pool causes — earners can climb while the rate still falls — and its logit is averaged with the direct fit's.
- A staleness adjustment. Exophase refreshes a game's counts on its own schedule, hours apart: when the tracked-player count has not moved for hours, the page is waiting on a refresh, and the pool that arrives with one dilutes most rates. The median shifts down a little and the spread widens by how overdue the refresh looks.
- The spread combines a random walk over the horizon (0.5 logit units per square-root day), the fit's own residuals, and the staleness term, so quiet trophies stay narrow and jumpy ones go wide.

## Validation

On the kit's two practice cases the program runs in about a tenth of a second, writes byte-identical output across runs, and passes a strict check of the output contract (one row per trophy, five non-decreasing quantiles in [0, 100], well under the size cap). The practice histories are the flat launch-night page, and the answers are the refreshed page (pool ×6.06, trophies earned ×4.53); against a reconstruction of that refresh consistent with the recorded aggregate counts, the program scores a pinball loss of about 0.13–0.14 against the persistence baseline's 0.20 on the same reconstruction.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=zbqpv30a&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
