"""GTA VI Twitch CCU forecaster (make.fun gta6-twitch-ccu-forecast-01).

Per-hour local-linear log regression over past big-launch comparables.

For each forecast hour h (2..24) we model the log ratio
    y_h = log(viewers_h / viewers_1)
across the comparables shipped inside the case file, regressing it on
    * log(hour-1 viewers) of the launch (scale),
    * the launch's pre-release trend (log slope of its hourly category
      viewers over the days before release),
    * the launch's pre-release level relative to its hour-1 print,
    * log viewers per live channel at hour 1,
    * a weekend indicator for the release day,
    * the launch's unlock shape: a simultaneous global unlock carries a
      step hour (parsed from the case notes for the target, read off the
      curve for a training launch), entered as hour-dependent
      post-step / hours-since-step features, plus an early-access flag.
The regression is weighted least squares with weights decaying by launch
dissimilarity (Gaussian kernel over the feature gaps to the target launch),
plus a small ridge.  The fitted value is the median log ratio; the
quantile spread around it comes from the per-hour regression residuals
(empirical residual quantiles, lightly smoothed across neighbouring hours
and shrunk toward a normal spread).  Everything is anchored to the case's
hour-1 print.  Pure Python 3.13 standard library, deterministic, no network.
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuning constants (chosen by leave-one-out CV over the comparables) ---
RIDGE = 1.0            # ridge penalty on the WLS normal equations
BW_LOGP1 = 1.0         # kernel bandwidth on |log p1 gap|
BW_TREND = 0.6         # kernel bandwidth on |trend gap| (log/day)
BW_PRELV = 1.2         # kernel bandwidth on |pre-release level gap|
SPREAD_SHRINK = 1.25   # shrink residual-quantile offsets toward the centre
SPREAD_BLEND = 0.65    # weight of empirical residual quantiles vs normal spread
TREND_DAYS = 3.0       # pre-release window (days) for trend / level features
MIN_TRAIN = 8          # minimum training launches for a per-hour regression
STEP_JUMP = 0.8        # min log jump between consecutive hours = an unlock step
STEP_SINCE_CAP = 8.0   # cap on hours-since-step feature
STEP_IN_REGRESSION = False  # include post-step / hours-since-step in the WLS design
PREDICT_MODE = "wls"   # "wls": local-linear regression; "localmedian": kernel-weighted median
REG_FEATURES = ("logp1", "trend", "pre_ratio", "vph", "weekend", "early")
USE_LOGP1_SQ = False   # add a squared logp1-gap term (curvature in the scale effect)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _ols_slope(xs, ys):
    """Slope of the least-squares line through (xs, ys)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def pre_release_features(points, day_start):
    """Trend slope (log viewers per day) and mean log level over the
    TREND_DAYS days before the release day, from hourly points."""
    xs, ys = [], []
    for pt in points:
        t = parse_time(pt["from"])
        gap_days = (day_start - t).total_seconds() / 86400.0
        if 0.0 < gap_days <= TREND_DAYS:
            v = pt.get("viewers")
            if v and v > 0:
                xs.append(-gap_days)
                ys.append(math.log(v))
    if not ys:
        return 0.0, None
    slope = _ols_slope(xs, ys)
    level = sum(ys) / len(ys)
    return slope, level


def launch_day_curve(hourly, day_start):
    """Map hour (1..24 of the UTC release day) -> (viewers, channels)."""
    curve = {}
    for pt in hourly:
        t = parse_time(pt["from"])
        dh = (t - day_start).total_seconds() / 3600.0
        if 0.0 <= dh < 24.0 and abs(dh - round(dh)) < 1e-6:
            h = int(round(dh)) + 1
            v = pt.get("viewers")
            if v is not None and v > 0:
                curve[h] = (v, pt.get("channels"))
    return curve


def detect_step(curve):
    """The hour of a launch's unlock step, from its own release-day curve:
    the hour with the largest log jump from the previous hour, when that
    jump is large (a simultaneous global unlock, not the rolling-midnight
    wave).  Returns the step hour (2..24) or None."""
    best_h, best_jump = None, STEP_JUMP
    for h in range(2, 25):
        if h in curve and (h - 1) in curve:
            jump = math.log(curve[h][0] / curve[h - 1][0])
            if jump > best_jump:
                best_h, best_jump = h, jump
    return best_h


def parse_unlock(text):
    """Parse a launch's unlock description (the case's notes for the target,
    a comparable's unlock field) into (is_global_step, step_hour).

    A single simultaneous global unlock carries a UTC time ("T02:00Z",
    "14:00Z"); a rolling midnight-local unlock has no single step hour.
    Returns (False, None) for rolling / unknown descriptions."""
    if not text:
        return False, None
    t = str(text).lower()
    globalish = (
        "single global" in t
        or "single simultaneous" in t
        or "simultaneous global" in t
        or "simultaneous across" in t
        or "simultaneous midnight" in t
        or "global unlock" in t
        or "unlock later" in t
    )
    if not globalish:
        return False, None
    times = re.findall(r"(?:t)?(\d{1,2}):(\d{2})\s*z", t)
    if not times:
        return True, None
    hh, _mm = times[0]
    hour = int(hh) + 1  # a 05:00Z unlock lands in hour 6 (05:00-06:00)
    return True, hour if 1 <= hour <= 24 else None


def features_for(p1, channels1, trend, pre_level, weekday):
    """Feature vector (without intercept) for one launch, centred so the
    target launch sits at the origin on the scale features."""
    logp1 = math.log(max(1.0, p1))
    pre_ratio = None
    if pre_level is not None:
        pre_ratio = pre_level - logp1
    vph = None
    if channels1 and channels1 > 0:
        vph = math.log(max(1.0, p1) / channels1)
    return {
        "logp1": logp1,
        "trend": trend,
        "pre_ratio": pre_ratio if pre_ratio is not None else 0.0,
        "vph": vph if vph is not None else 0.0,
        "weekend": 1.0 if weekday >= 5 else 0.0,
        "step": None,       # step hour (2..24) or None; set by caller
        "early": 0.0,       # 1 when the launch had a paid early-access period
    }


def dissimilarity(fa, fb):
    """Distance between two launches' feature dicts (target vs comparable)."""
    d2 = ((fa["logp1"] - fb["logp1"]) / BW_LOGP1) ** 2
    d2 += ((fa["trend"] - fb["trend"]) / BW_TREND) ** 2
    d2 += ((fa["pre_ratio"] - fb["pre_ratio"]) / BW_PRELV) ** 2
    d2 += 0.25 * ((fa["vph"] - fb["vph"])) ** 2
    d2 += 0.5 * ((fa["weekend"] - fb["weekend"])) ** 2
    d2 += 0.8 * ((fa["early"] - fb["early"])) ** 2
    sa, sb = fa["step"], fb["step"]
    if (sa is None) != (sb is None):
        d2 += 1.2 ** 2
    elif sa is not None and sb is not None:
        d2 += (0.3 * abs(sa - sb) / 6.0) ** 2
    return math.sqrt(d2)


def _step_feats(feat, hour):
    """Hour-dependent unlock-step features for one launch at one hour:
    (post_step, hours_since_step capped).  Rolling launches get (0, 0)."""
    u = feat["step"]
    if u is None or hour < u:
        return 0.0, 0.0
    return 1.0, min(float(hour - u), STEP_SINCE_CAP)


def solve_linear(a, b):
    """Solve a x = b for a small dense symmetric system (Gaussian elim)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            m[piv][col] = 1e-12
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                fac = m[r][col] / pv
                for cc in range(col, n + 1):
                    m[r][cc] -= fac * m[col][cc]
    return [m[i][n] / m[i][i] for i in range(n)]


def wls_predict(train, target_feat, hour):
    """Weighted least squares for one hour (or kernel-weighted local median
    when PREDICT_MODE == "localmedian").

    train: list of (feat dict, y, weight) with y = log ratio for this hour.
    Returns (fitted median log ratio, residual list, weights) or None when
    too little training data.
    """
    if len(train) < MIN_TRAIN:
        return None
    if PREDICT_MODE == "localmedian":
        pairs = [(y, w) for _f, y, w in train]
        fitted = weighted_quantile(pairs, 0.5)
        residuals, weights = [], []
        for f, y, w in train:
            # leave-one-out local median at this launch, for an honest spread
            others = [(y2, w2) for f2, y2, w2 in train if f2 is not f]
            pred = weighted_quantile(others, 0.5)
            residuals.append(y - pred)
            weights.append(w)
        return fitted, residuals, weights
    # design: [1] + one gap column per REG_FEATURE (+ squared logp1 gap,
    # + step columns when enabled)
    def row(f, hour):
        base = [1.0] + [f[key] - target_feat[key] for key in REG_FEATURES]
        if USE_LOGP1_SQ:
            gap = f["logp1"] - target_feat["logp1"]
            base.append(gap * gap)
        if not STEP_IN_REGRESSION:
            return base
        post, since = _step_feats(f, hour)
        tpost, tsince = _step_feats(target_feat, hour)
        return base + [post - tpost, since - tsince]

    k = 1 + len(REG_FEATURES) + (1 if USE_LOGP1_SQ else 0) + (2 if STEP_IN_REGRESSION else 0)
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for f, y, w in train:
        r = row(f, hour)
        for i in range(k):
            atb[i] += w * r[i] * y
            for j in range(k):
                ata[i][j] += w * r[i] * r[j]
    for i in range(k):
        ata[i][i] += RIDGE if i else 1e-9
    beta = solve_linear(ata, atb)
    # fitted value at the target (all gaps are zero) is the intercept
    fitted = beta[0]
    residuals = []
    weights = []
    for f, y, w in train:
        r = row(f, hour)
        pred = sum(b * x for b, x in zip(beta, r))
        residuals.append(y - pred)
        weights.append(w)
    return fitted, residuals, weights


def weighted_quantile(pairs, tau):
    """Weighted empirical quantile of (value, weight) pairs."""
    if not pairs:
        return 0.0
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    cut = tau * total
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= cut:
            return v
    return pairs[-1][0]


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    known1 = next((k for k in case["known"] if k["hour"] == 1), None)
    if known1 is None:
        raise ValueError("no hour-1 print")
    p1 = max(1.0, float(known1["viewers"]))
    day_start = parse_time(case["day_start"])
    weekday = day_start.weekday()

    trend, pre_level = pre_release_features(case.get("history_hourly") or [], day_start)
    target_feat = features_for(p1, known1.get("channels"), trend, pre_level, weekday)
    # The target's unlock shape comes from its notes (rolling midnight for
    # GTA VI and every practice launch); a simultaneous global unlock would
    # carry its UTC hour there.
    _is_global, target_step = parse_unlock(case.get("notes") or "")
    target_feat["step"] = target_step if _is_global else None
    notes_low = str(case.get("notes") or "").lower()
    target_feat["early"] = 1.0 if "early access" in notes_low else 0.0

    # Build the training set from the comparables.
    train_by_hour = {h: [] for h in range(2, 25)}
    pooled_ratios = {h: [] for h in range(2, 25)}
    for comp in case.get("comparables") or []:
        curve = launch_day_curve(comp.get("hourly") or [], parse_time(comp["day_start"]))
        if 1 not in curve:
            continue
        cp1, cch1 = curve[1]
        ctrend, cpre = pre_release_features(comp.get("hourly") or [], parse_time(comp["day_start"]))
        cfeat = features_for(cp1, cch1, ctrend, cpre, parse_time(comp["day_start"]).weekday())
        # A training launch's unlock step is read off its own curve (its
        # answer is training data); its early-access flag off its unlock text.
        cfeat["step"] = detect_step(curve)
        unlock_low = str(comp.get("unlock") or "").lower()
        cfeat["early"] = 1.0 if ("early access" in unlock_low or "early-access" in unlock_low) else 0.0
        d = dissimilarity(cfeat, target_feat)
        w = math.exp(-0.5 * d * d)
        for h in range(2, 25):
            if h in curve:
                y = math.log(curve[h][0] / cp1)
                train_by_hour[h].append((cfeat, y, w))
                pooled_ratios[h].append(y)

    hours_out = []
    # residual quantiles per hour, smoothed across neighbouring hours
    raw_q = {}
    for h in range(2, 25):
        res = wls_predict(train_by_hour[h], target_feat, h)
        if res is None:
            # fall back to the pooled median log ratio and a wide spread
            ratios = sorted(pooled_ratios[h]) or [0.0]
            med = ratios[len(ratios) // 2]
            raw_q[h] = (med, [-0.9, -0.4, 0.0, 0.4, 0.9])
            continue
        fitted, residuals, weights = res
        pairs = list(zip(residuals, weights))
        # centre the fitted median on the residual median
        med_res = weighted_quantile(pairs, 0.5)
        fitted += med_res
        emp = [weighted_quantile(pairs, tau) - med_res for tau in LEVELS]
        # normal-spread blend: residual std -> z * sigma
        wsum = sum(weights)
        mean_r = sum(r * w for r, w in pairs) / wsum if wsum else 0.0
        var = sum(w * (r - mean_r) ** 2 for r, w in pairs) / wsum if wsum else 0.25
        sigma = math.sqrt(max(var, 0.01))
        norm = [z * sigma for z in Z]
        spread = [
            SPREAD_BLEND * e + (1.0 - SPREAD_BLEND) * n for e, n in zip(emp, norm)
        ]
        spread = [s * SPREAD_SHRINK for s in spread]
        # keep the median offset at 0 and the spread monotone
        spread[2] = 0.0
        for i in range(1, 5):
            spread[i] = max(spread[i], spread[i - 1])
        for i in range(3, -1, -1):
            spread[i] = min(spread[i], spread[i + 1])
        raw_q[h] = (fitted, spread)

    # light smoothing of the fitted medians and spreads across hours
    for h in range(2, 25):
        neigh = [hh for hh in (h - 1, h, h + 1) if 2 <= hh <= 24]
        med = sum(raw_q[hh][0] for hh in neigh) / len(neigh)
        # keep the hour's own value dominant: 85% own, 15% neighbours, so a
        # genuine unlock step in the fitted curve is not blurred away
        med = 0.85 * raw_q[h][0] + 0.15 * med
        spread = []
        for i in range(5):
            vals = [raw_q[hh][1][i] for hh in neigh]
            spread.append(sum(vals) / len(vals))
        spread[2] = 0.0
        for i in range(1, 5):
            spread[i] = max(spread[i], spread[i - 1] + 1e-9)
        for i in range(3, -1, -1):
            spread[i] = min(spread[i], spread[i + 1] - 1e-9)
        q = [p1 * math.exp(med + s) for s in spread]
        q = [min(max(v, 1.0), 1e9) for v in q]
        q = [round(v, 2) for v in q]
        # enforce non-decreasing after rounding
        for i in range(1, 5):
            q[i] = max(q[i], q[i - 1])
        hours_out.append({"hour": h, "q": q})
    return {"hours": hours_out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
