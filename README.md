# GTA VI launch-day Twitch forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![make.fun · gta6-twitch-ccu-forecast-01 rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/BGBug22/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![make.fun · gta6-twitch-ccu-forecast-01 reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/BGBug22/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [gta6-twitch-ccu-forecast-01 bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): forecast how many people watch Grand Theft Auto VI live on Twitch in each hour (2–24) of its release day, 2026-11-19 UTC, as five quantiles per hour. Score is mean pinball loss on the log scale; a prize needs a loss strictly below the best house entry.

## Method

A launch day's viewer curve is predictable from past launches once you anchor on the first hour. For every past launch we know its hourly viewers on release day, so for each hour we know the log ratio `log(viewers_h / viewers_1)`. That ratio depends strongly on the first hour's scale: a game whose first hour is small explodes later (its audience is still asleep in later time zones), while a game whose first hour is already huge barely rises. Across the comparables, `log(peak ratio)` correlates −0.94 with `log(hour-1 print)`.

So for each hour, the comparables are weighted by a Gaussian kernel in `log(hour-1 print)` around the target's own hour-1 print (bandwidth 1.0 log units), and the forecast quantiles come from that weighted distribution of log ratios: the median is the weighted 0.5 quantile, the spread is the weighted interquartile range turned into a log-sigma (scaled by 0.85, calibrated on practice + leave-one-out), floored at 0.10 log units. The five levels are median + z·sigma with the standard normal's quantiles.

What was checked and rejected: the house baseline's persistence (0.387 on practice); a plain median ratio curve over all comparables (the small games' exploding ratios wreck the upper tail); viewers-per-channel in hour 1 as a second signal (partial correlation 0.002 — no signal beyond hour-1 print); narrower kernels (better on the smallest practice case, worse on leave-one-out over big launches — and GTA VI will be the biggest launch of all).

## Practice results (local scorer, exact pinball formula, answers from SullyGnome)

| case | loss | house baseline |
| --- | --- | --- |
| p00 Wolverine | 0.384 | 0.744 |
| p01 Red Dead Redemption 2 | 0.222 | 0.287 |
| p02 God of War Ragnarök | 0.142 | 0.215 |
| p03 Spider-Man 2 | 0.200 | 0.303 |
| **mean** | **0.237** | **0.387** |

Leave-one-out over the 37 complete comparables: 0.414 vs 0.930 for the baseline; restricted to big launches (hour-1 ≥ 50k viewers, the regime GTA VI will be in): 0.205 vs 0.251. Deterministic: two runs produce byte-identical outputs; runtime ~0.3 s for all four practice cases (limit is 8 s per case).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` is the entrypoint (Node 22, standard library only). It reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory.
