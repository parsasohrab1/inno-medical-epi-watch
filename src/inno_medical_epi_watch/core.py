"""Syndromic surveillance: outbreak detection, short-term forecast and provincial report.

Lab prototype on SYNTHETIC data only (TRL 4).
"""
import numpy as np

PROVINCES = ["P%02d" % i for i in range(12)]
DISEASES = ["influenza_like", "gastroenteritis", "measles"]
WEEKS = 260
Z = 4.0

CRITERIA = [
    ("outbreak_recall", ">=", 0.80, "share of injected outbreaks detected"),
    ("false_alarm_rate", "<=", 0.01, "false alerts per non-outbreak series-week"),
    ("forecast_gain", ">=", 0.10, "forecast MAE improvement over 4-week average"),
]


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(WEEKS)
    base = {"influenza_like": 60, "gastroenteritis": 40, "measles": 6}
    amp = {"influenza_like": 0.7, "gastroenteritis": 0.4, "measles": 0.3}
    peak = {"influenza_like": 3, "gastroenteritis": 30, "measles": 12}
    S = len(PROVINCES) * len(DISEASES)
    lam = np.zeros((S, WEEKS)); keys = []
    k = 0
    for p in PROVINCES:
        size = rng.uniform(0.5, 2.0)
        for d in DISEASES:
            lam[k] = size * base[d] * (1 + amp[d] * np.cos(2 * np.pi * (t - peak[d]) / 52))
            keys.append((p, d)); k += 1
    y = rng.poisson(lam)
    outbreaks = []
    for _ in range(14):
        s = int(rng.integers(0, S)); start = int(rng.integers(160, WEEKS - 6)); dur = int(rng.integers(3, 6))
        if any(o["series"] == s and abs(o["start"] - start) < 12 for o in outbreaks):
            continue
        add = lam[s, start:start + dur] * rng.uniform(1.2, 2.5) + 6
        y[s, start:start + dur] += rng.poisson(add)
        outbreaks.append({"series": s, "start": start, "end": start + dur - 1})
    return keys, y, outbreaks


def expected(series, t):
    """Seasonal expectation for week t from the same weeks (+-2) of the previous 3 years."""
    vals = [series[t - 52 * k + j] for k in (1, 2, 3) for j in range(-2, 3) if 0 <= t - 52 * k + j < t]
    return float(np.mean(vals)) if len(vals) >= 5 else None


def detect(series, z=Z, start=156):
    alerts = []
    for t in range(start, len(series)):
        mu = expected(series, t)
        if mu is None:
            continue
        thr = mu + z * np.sqrt(mu + 1.0)
        if series[t] > thr:
            alerts.append({"week": t, "observed": int(series[t]), "expected": round(mu, 1), "threshold": round(thr, 1)})
    return alerts


def forecast(series, horizon=4):
    t0 = len(series)
    exp = np.array([expected(np.concatenate([series, np.zeros(horizon)]), t0 + h) or series[-4:].mean() for h in range(horizon)])
    recent = series[-4:].sum()
    ref = np.array([expected(series, t0 - 4 + j) or series[t0 - 4 + j] for j in range(4)]).sum()
    ratio = np.clip(recent / max(ref, 1.0), 0.5, 2.0)
    return exp * (0.5 + 0.5 * ratio)


def provincial_report(keys, y, week):
    lines = ["Weekly report, week %d" % week]
    for s, (p, d) in enumerate(keys):
        a = [x for x in detect(y[s]) if x["week"] == week]
        if a:
            lines.append("ALERT %s %s: observed %d vs expected %.1f" % (p, d, a[0]["observed"], a[0]["expected"]))
    if len(lines) == 1:
        lines.append("No alerts.")
    return "\n".join(lines)


def evaluate(seed=0):
    keys, y, outs = simulate(seed)
    inwin = np.zeros_like(y, bool)
    for o in outs:
        inwin[o["series"], o["start"]:o["end"] + 1] = True
    alerts = [detect(y[s]) for s in range(len(keys))]
    hit = sum(any(o["start"] <= a["week"] <= o["end"] for a in alerts[o["series"]]) for o in outs)
    fa = sum(1 for s in range(len(keys)) for a in alerts[s] if not inwin[s, a["week"]])
    denom = int((~inwin[:, 156:]).sum())
    errs_m, errs_b = [], []
    for s in range(len(keys)):
        for T in range(170, WEEKS - 4, 6):
            if inwin[s, T - 8:T + 4].any():
                continue
            f = forecast(y[s, :T]); a = y[s, T:T + 4]
            errs_m.append(np.abs(f - a).mean()); errs_b.append(np.abs(y[s, T - 4:T].mean() - a).mean())
    return {"outbreak_recall": hit / max(len(outs), 1), "false_alarm_rate": fa / denom,
            "forecast_gain": 1 - np.mean(errs_m) / np.mean(errs_b), "n_outbreaks": len(outs)}


def generate(seed=0):
    keys, y, outs = simulate(seed)
    rows = [dict(province=p, disease=d, week=int(t), cases=int(y[s, t])) for s, (p, d) in enumerate(keys) for t in range(WEEKS)]
    labels = [dict(province=keys[o["series"]][0], disease=keys[o["series"]][1], start_week=o["start"], end_week=o["end"]) for o in outs]
    return {"weekly_cases": rows, "injected_outbreaks": labels}


def demo(seed=0):
    keys, y, outs = simulate(seed)
    o = outs[0]
    return provincial_report(keys, y, o["start"] + 1)
