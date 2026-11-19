"""Lapak-forecast entry for make.fun's gta6-trophy-rarity-forecast-01 (Python 3.13, standard library only).

The program reads one case file per run: every read the house recorded of the GTA VI trophy page on
Exophase before the case's checkpoint, and writes five quantiles (5/25/50/75/95%) of each trophy's earn
rate at the case's target. Scoring is mean pinball loss on the logit scale, so all modelling happens on
the logit scale.

Model, per trophy:
  - a recency-weighted least-squares line of logit(rate) against ln(1 + hours since release), read at
    the target; log-time makes the extrapolation saturate the way trophy curves do, and each slope is
    shrunk toward its tier's pooled slope by the fit's effective sample size;
  - a second line of the same shape fitted to the trophy's earner count (rate x players tracked) and one
    to the players-tracked count itself; their ratio at the target is a second forecast that sees the
    dilution a growing tracked pool causes, and the two logits are averaged;
  - a staleness adjustment: when the tracked-player count has not moved for hours, the page is waiting
    on an Exophase refresh, and at a launch the pool that arrives with a refresh dilutes most rates, so
    the median shifts down a little and the spread widens by how overdue the refresh looks;
  - a spread that combines a random walk over the horizon, the fit's own residuals and the staleness
    term, so quiet trophies stay narrow and jumpy ones go wide.

The program is deterministic: two runs on the same case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

# The grader runs python3 -P, which keeps this file's directory off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level above.
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
# Release day begins 2026-11-19T00:00:00Z; curve time is ln(1 + hours since then).
ANCHOR = datetime(2026, 11, 19, tzinfo=timezone.utc)
# Rates are clamped to the range the scorer clamps to before it takes logits.
RATE_LO = 0.01
RATE_HI = 99.99
# Recency weighting of the fits: a read this many hours before the last one counts half.
HALF_LIFE_HOURS = 36.0
# Random-walk spread of a rate's logit per square-root day of horizon.
RW_SIGMA = 0.5
# How far an overdue refresh shifts a stale page's medians down, in logit units.
STALE_SHIFT = -0.35
# Extra spread, in logit units, for a page that looks fully overdue for a refresh.
STALE_SPREAD = 0.85
# Hours without a players-tracked change after which a refresh counts as fully overdue.
STALE_FULL_HOURS = 4.0
# The narrowest spread the program will ever report, in logit units.
SIGMA_FLOOR = 0.12


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def curve_x(moment):
    hours = (moment - ANCHOR).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def clamp_rate(rate):
    return min(RATE_HI, max(RATE_LO, rate))


def logit_pct(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    return 100.0 / (1.0 + math.exp(-value))


def round4(value):
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 10000.0
    return int(result) if float(result).is_integer() else result


def z_for_level(level):
    for known, z in zip(LEVELS, Z):
        if abs(level - known) < 1e-9:
            return z
    return z_approx(level)


def z_approx(p):
    # Acklam's rational approximation of the normal quantile, for levels outside the scored five.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > 0.97575:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den
    q = p - 0.5
    r = q * q
    num = ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    return num * q / den


def wls(points):
    """Weighted least squares of y on x for points (x, y, w): (intercept, slope, residual sd, n_eff)."""
    total_w = sum(w for _, _, w in points)
    if total_w <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    mean_x = sum(w * x for x, _, w in points) / total_w
    mean_y = sum(w * y for _, y, w in points) / total_w
    var_x = sum(w * (x - mean_x) ** 2 for x, _, w in points) / total_w
    n_eff = total_w * total_w / sum(w * w for _, _, w in points)
    if var_x < 1e-9:
        resid = math.sqrt(sum(w * (y - mean_y) ** 2 for _, y, w in points) / total_w)
        return mean_y, 0.0, resid, n_eff
    cov = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in points) / total_w
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    resid = math.sqrt(sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in points) / total_w)
    return intercept, slope, resid, n_eff


def recency_weight(moment, last_moment):
    age_hours = (last_moment - moment).total_seconds() / 3600.0
    return 0.5 ** (age_hours / HALF_LIFE_HOURS)


def prepare_reads(case):
    reads = []
    for entry in case["history"]:
        moment = parse_time(entry.get("at") or entry["slot"])
        reads.append((moment, entry))
    reads.sort(key=lambda pair: pair[0])
    return reads


def staleness(reads):
    """Hours since the players-tracked count last changed, and the span of the whole history."""
    if not reads:
        return 0.0, 0.0
    last_moment = reads[-1][0]
    span_hours = (last_moment - reads[0][0]).total_seconds() / 3600.0
    last_players = reads[-1][1].get("players_tracked")
    stale_since = reads[0][0]
    for moment, entry in reads:
        if entry.get("players_tracked") != last_players:
            stale_since = moment
    stale_hours = (last_moment - stale_since).total_seconds() / 3600.0
    if all(entry.get("players_tracked") == last_players for _, entry in reads):
        stale_hours = span_hours
    return stale_hours, span_hours


def forecast_case(case):
    trophies = case["trophies"]
    levels = case.get("levels") or LEVELS
    zs = [z_for_level(level) for level in levels]
    reads = prepare_reads(case)
    if not reads:
        raise ValueError("case has no history")
    last_moment = reads[-1][0]
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"]) if case.get("target") else checkpoint + timedelta(minutes=case["horizon_minutes"])
    x_target = curve_x(target)
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0
    weights = [recency_weight(moment, last_moment) for moment, _ in reads]
    xs = [curve_x(moment) for moment, _ in reads]
    players = [entry.get("players_tracked") for _, entry in reads]

    # The players-tracked count's own curve, for the earner-ratio forecast.
    pool_fit = None
    if all(isinstance(n, (int, float)) and n > 0 for n in players):
        pool_points = [(x, math.log(n), w) for x, n, w in zip(xs, players, weights)]
        pool_fit = wls(pool_points)

    # First pass: each trophy's direct logit fit, kept for the tier pools.
    direct = []
    for i in range(len(trophies)):
        points = []
        last_rate = None
        for (moment, entry), x, w in zip(reads, xs, weights):
            rate = entry["rates"][i] if i < len(entry.get("rates") or []) else None
            if rate is None:
                continue
            points.append((x, logit_pct(rate), w))
            last_rate = rate
        fit = wls(points) if len(points) >= 2 else None
        direct.append((points, fit, last_rate))

    tier_slopes = {}
    for trophy, (_, fit, _) in zip(trophies, direct):
        if fit is None:
            continue
        tier_slopes.setdefault(trophy.get("tier", "?"), []).append(fit[1])
    tier_median_slope = {}
    for tier, slopes in tier_slopes.items():
        ordered = sorted(slopes)
        tier_median_slope[tier] = ordered[len(ordered) // 2]

    stale_hours, _ = staleness(reads)
    stale_factor = min(1.0, stale_hours / STALE_FULL_HOURS)

    rows = []
    for i, trophy in enumerate(trophies):
        points, fit, last_rate = direct[i]
        rows.append(forecast_trophy(i, trophy, points, fit, last_rate, reads, xs, weights, pool_fit, tier_median_slope, x_target, horizon_days, stale_factor, zs))
    return {"trophies": rows}


def forecast_trophy(index, trophy, points, fit, last_rate, reads, xs, weights, pool_fit, tier_median_slope, x_target, horizon_days, stale_factor, zs):
    if last_rate is None:
        # No read ever listed this trophy: sit at the clamp floor with a wide, honest spread.
        centre = logit_pct(RATE_LO)
        sigma = 1.5
        return {"id": trophy["id"], "q": quantiles(centre, sigma, zs)}
    centre_direct = None
    resid = 0.0
    if fit is not None:
        intercept, slope, resid, n_eff = fit
        pooled = tier_median_slope.get(trophy.get("tier", "?"), slope)
        trust = n_eff / (n_eff + 10.0)
        slope = trust * slope + (1.0 - trust) * pooled
        centre_direct = intercept + slope * x_target
    centre_ratio = None
    if pool_fit is not None:
        centre_ratio = ratio_centre(index, reads, xs, weights, pool_fit, x_target)
    centres = [c for c in (centre_direct, centre_ratio) if c is not None]
    if centres:
        centre = sum(centres) / len(centres)
    else:
        centre = logit_pct(last_rate)
    if stale_factor > 0.0 and last_rate < 99.9:
        centre += STALE_SHIFT * stale_factor
    centre = min(logit_pct(RATE_HI), max(logit_pct(RATE_LO), centre))
    walk = RW_SIGMA * math.sqrt(horizon_days)
    resid_term = min(0.8 * resid, 1.2)
    stale_term = STALE_SPREAD * stale_factor
    sigma = math.sqrt(walk * walk + resid_term * resid_term + stale_term * stale_term + 0.15 ** 2)
    sigma = max(sigma, SIGMA_FLOOR)
    return {"id": trophy["id"], "q": quantiles(centre, sigma, zs)}


def ratio_centre(index, reads, xs, weights, pool_fit, x_target):
    """Forecast logit from the earner count's curve divided by the tracked pool's curve."""
    points = []
    for (_, entry), x, w in zip(reads, xs, weights):
        rate = entry["rates"][index] if index < len(entry.get("rates") or []) else None
        players = entry.get("players_tracked")
        if rate is None or not isinstance(players, (int, float)) or players <= 0:
            continue
        earners = rate * players / 100.0
        points.append((x, math.log(max(earners, 0.3)), w))
    if len(points) < 2:
        return None
    intercept, slope, _, _ = wls(points)
    log_earners = intercept + slope * x_target
    log_players = pool_fit[0] + pool_fit[1] * x_target
    log_share = min(0.0, max(-12.0, log_earners - log_players))
    return logit_pct(100.0 * math.exp(log_share))


def quantiles(centre, sigma, zs):
    values = []
    previous = None
    for z in zs:
        value = round4(inv_logit_pct(centre + z * sigma))
        if previous is not None and value < previous:
            value = previous
        values.append(value)
        previous = value
    return values


def persistence_case(case):
    """Fallback if the model ever throws: last rate, random-walk spread."""
    trophies = case["trophies"]
    levels = case.get("levels") or LEVELS
    zs = [z_for_level(level) for level in levels]
    reads = prepare_reads(case)
    last_entry = reads[-1][1] if reads else {"rates": []}
    horizon_days = max(case.get("horizon_minutes", 0), 0) / 1440.0
    sigma = max(RW_SIGMA * math.sqrt(horizon_days), SIGMA_FLOOR)
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_entry["rates"][i] if i < len(last_entry.get("rates") or []) else None
        centre = logit_pct(rate) if rate is not None else logit_pct(RATE_LO)
        rows.append({"id": trophy["id"], "q": quantiles(centre, sigma, zs)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = persistence_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
