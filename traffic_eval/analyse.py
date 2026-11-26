#!/usr/bin/env python3
"""Score the three ways of using traffic incidents against what the buses actually did.

    analyse.py            reads work/q_*.npz + work/m_*.npz (+ results/gps_*_passes.csv), writes results/

Questions, per bus system, on held-out days only (days the v4 models did not train on):
  1. How many queries have an active incident on the bus's remaining path, from how many distinct incidents?
  2. On those queries, is v4 late (the bus arrives later than predicted) beyond what the same route, stop and hour
     show without an incident? (difference-in-differences: treated residual minus the matched control residual,
     control cell = same route x target stop x IST hour on any evaluated day without an incident on the path)
  3. Route (b): does serve/incident_penalty.py's uncalibrated delay help or hurt (MAE with and without it)?
     Calibrated variant: delay per incident type = the median excess lateness of that type on the OTHER days
     (leave-one-day-out), added only where positive.
  4. Route (a): a residual LightGBM on top of v4 with incident features (has incident, type, severity, metres to it,
     minutes since its start, horizon), trained leave-one-day-out on that system's own days.
  5. Route (c), the GPS: from results/gps_<SYSTEM>_passes.csv.gz, bus speed at the incident vs the same spot and time
     of day on other days, against minutes since the alert.
Confidence intervals: 2,000 bootstrap resamples of INCIDENTS (queries of one incident are not independent).
"""
from __future__ import annotations

import collections
import json

import numpy as np

from common import RESULTS, WORK, TRAIN_DAYS, day_bounds

RNG = np.random.default_rng(7)
TYPES = ["accident", "breakdown", "closure", "congestion", "diversion", "event", "fire", "other", "spillage",
         "tree_fall", "waterlogging", "works", "u_turn_change"]


def load(system, events):
    parts = []
    for q in sorted(WORK.glob(f"q_{system}_*.npz")):
        day = q.stem.split("_")[2]
        if day in TRAIN_DAYS.get(system, []):
            continue
        m = WORK / q.name.replace("q_", "m_", 1)
        if not m.exists():
            continue
        z, w = np.load(q), np.load(m)
        if "y" not in z.files or not len(z["y"]):
            continue
        d = {c: z[c] for c in ("reg", "code", "t", "s", "k", "dk", "y", "b", "p", "lo", "hi")}
        d.update({c: w[c] for c in w.files})
        for tag in ("pre", "post"):
            mp = WORK / q.name.replace("q_", f"m_{tag}_", 1)
            if mp.exists():
                wp = np.load(mp)
                d[f"n_inc_{tag}"], d[f"inc_{tag}"] = wp["n_inc"], wp["inc"]
            else:
                d[f"n_inc_{tag}"], d[f"inc_{tag}"] = np.zeros(len(d["y"]), np.int16), np.full(len(d["y"]), -1, np.int32)
        d["day"] = np.full(len(d["y"]), day)
        parts.append(d)
    if not parts:
        return None
    D = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
    D["hour"] = ((D["t"] + 19800) // 3600 % 24).astype(int)
    D["typ"] = np.array([events[i]["type"] if i >= 0 else "" for i in D["inc"]])
    D["sev"] = np.array([events[i]["severity"] if i >= 0 else 0.0 for i in D["inc"]])
    return D


def mae(x):
    return float(np.mean(np.abs(x))) if len(x) else float("nan")


def boot(values_by_cluster, stat=np.mean, n=2000):
    """CI of stat(all values) resampling clusters (incidents)."""
    keys = list(values_by_cluster)
    if len(keys) < 2:
        return [float("nan"), float("nan")]
    arrs = [np.asarray(values_by_cluster[k]) for k in keys]
    out = []
    for _ in range(n):
        pick = RNG.integers(0, len(keys), len(keys))
        out.append(stat(np.concatenate([arrs[i] for i in pick])))
    return [round(float(np.percentile(out, 2.5)), 1), round(float(np.percentile(out, 97.5)), 1)]


def boot_iw(values_by_cluster, n=2000):
    """Incident-weighted mean (mean over incidents of each incident's mean) and its CI, resampling incidents."""
    keys = list(values_by_cluster)
    m = np.array([np.mean(values_by_cluster[k]) for k in keys])
    if len(m) < 2:
        return round(float(m.mean()), 1) if len(m) else float("nan"), [float("nan"), float("nan")]
    bs = [m[RNG.integers(0, len(m), len(m))].mean() for _ in range(n)]
    return round(float(m.mean()), 1), [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]


def clusters(inc, vals):
    c = collections.defaultdict(list)
    for e_, x in zip(inc, vals):
        c[int(e_)].append(float(x))
    return c


def control_means(D, ctrl):
    """Mean lateness (y - p) and nowcast lateness (y - b) of the control queries per cell, with fallbacks."""
    cells = []
    for keyf in (lambda i: (D["code"][i], int(D["k"][i]), int(D["hour"][i])),
                 lambda i: (D["code"][i], int(D["hour"][i])),
                 lambda i: (int(D["hour"][i]), int(min(D["y"][i], 1799) // 300))):
        acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
        for i in np.flatnonzero(ctrl):
            a = acc[keyf(i)]
            a[0] += D["y"][i] - D["p"][i]; a[1] += D["y"][i] - D["b"][i]; a[2] += 1
        cells.append((keyf, {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in acc.items() if v[2] >= 5}))
    return cells


def matched(D, idx, cells):
    late_c, late_bc, level = np.full(len(idx), np.nan), np.full(len(idx), np.nan), np.zeros(len(idx), int)
    for j, i in enumerate(idx):
        for lv, (keyf, tab) in enumerate(cells):
            v = tab.get(keyf(i))
            if v:
                late_c[j], late_bc[j], level[j] = v[0], v[1], lv
                break
    return late_c, late_bc, level


def residual_model(D, tr_mask_all, days, with_incident=True):
    """Route (a): leave-one-day-out residual LightGBM on top of v4. with_incident=False is the ablation: the same
    model without the five incident features, so any gain it shows is NOT the incidents' (e.g. v4's general bias
    on a system it never trained on). Returns the adjusted P50 for every query of every held-out day."""
    import lightgbm as lgb
    tix = {t: n for n, t in enumerate(TYPES)}
    base = [D["p"], D["dk"] - D["s"], D["hour"].astype(float)]
    inc = [(D["n_inc"] > 0).astype(float), np.array([tix.get(t, np.nan) for t in D["typ"]], float), D["sev"],
           np.nan_to_num(D["d_inc"], nan=-1.0), np.nan_to_num(D["age_min"], nan=-1.0)]
    X = np.column_stack((inc + base) if with_incident else base)
    target = np.log(np.clip(D["y"], 30, None) / np.clip(D["p"], 30, None))
    adj = D["p"].copy()
    for d in days:
        te = D["day"] == d
        tr = ~te
        # the treated queries plus 4x as many of the rest, so the model sees what "no incident" means
        tri = np.flatnonzero(tr & tr_mask_all)
        oth = np.flatnonzero(tr & ~tr_mask_all)
        if len(tri) < 200:
            continue
        oth = RNG.choice(oth, min(len(oth), 4 * len(tri)), replace=False)
        ii = np.concatenate([tri, oth])
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                              objective="huber", alpha=0.3, verbose=-1)
        m.fit(X[ii], target[ii], categorical_feature=[1] if with_incident else "auto")
        tei = np.flatnonzero(te)
        adj[tei] = D["p"][tei] * np.exp(m.predict(X[tei]))
    return adj


def gps_summary(system):
    import csv
    import gzip
    p = RESULTS / f"gps_{system}_passes.csv.gz"
    if not p.exists():
        return None
    rows = list(csv.DictReader(gzip.open(p, "rt")))
    if not rows or "ratio" not in rows[0]:
        return None
    bins = [(-120, -60), (-60, -30), (-30, 0), (0, 15), (15, 30), (30, 60), (60, 120), (120, 240)]
    real = [r for r in rows if r.get("placebo", "0") == "0"]
    plac = [r for r in rows if r.get("placebo", "0") == "1"]
    out = {"n_passes": len(real), "n_events": len({r["event"] for r in real}), "n_placebo_passes": len(plac),
           "by_minutes_since_alert": {}, "placebo_by_minutes": {}}
    for name, rs_ in (("by_minutes_since_alert", real), ("placebo_by_minutes", plac)):
        for lo, hi in bins:
            sel = [r for r in rs_ if lo <= float(r["tau_min"]) < hi]
            if not sel:
                continue
            evs = collections.defaultdict(list)
            for r in sel:
                evs[r["event"] + ("/" + r.get("day", "") if name != "by_minutes_since_alert" else "")].append(float(r["ratio"]))
            med = float(np.median([float(r["ratio"]) for r in sel]))
            out[name][f"{lo}..{hi}"] = {"passes": len(sel), "events": len(evs), "median_speed_ratio": round(med, 3),
                                        "ci95": [x / 100 for x in boot({k: [v * 100 for v in vs] for k, vs in evs.items()}, np.median, 500)]}
    rows = real
    by_ev = collections.defaultdict(list)
    for r in rows:
        by_ev[r["event"]].append(r)
    # per event: was it slow during the window at all, and when did the first slow pass after the alert come?
    slow, lag, n_ev = 0, [], 0
    for ev, rs in by_ev.items():
        dur = float(rs[0]["dur_min"])
        inwin = [r for r in rs if 0 <= float(r["tau_min"]) <= min(dur, 120)]
        if len(inwin) < 3:
            continue
        n_ev += 1
        if np.median([float(r["ratio"]) for r in inwin]) < 0.8:
            slow += 1
        pre = [r for r in rs if -60 <= float(r["tau_min"]) < 0]
        first = sorted(float(r["tau_min"]) for r in inwin if float(r["ratio"]) < 0.7)
        if first:
            lag.append(first[0])
        out.setdefault("pre_alert_slow_events", 0)
        if pre and np.median([float(r["ratio"]) for r in pre]) < 0.8:
            out["pre_alert_slow_events"] += 1
    # the same rule on the placebo days: how often is a spot that slow at that clock time anyway?
    pb = collections.defaultdict(list)
    for r in plac:
        pb[(r["event"], r.get("day"))].append(r)
    p_n = p_slow = 0
    for rs in pb.values():
        dur = float(rs[0]["dur_min"])
        inwin = [float(r["ratio"]) for r in rs if 0 <= float(r["tau_min"]) <= min(dur, 120)]
        if len(inwin) >= 3:
            p_n += 1
            p_slow += np.median(inwin) < 0.8
    out["placebo_spot_days_with_3plus_passes"] = p_n
    out["placebo_spot_days_slow(median ratio<0.8)"] = int(p_slow)
    out["events_with_3plus_passes_in_window"] = n_ev
    out["events_slow_in_window(median ratio<0.8)"] = slow
    out["minutes_to_first_slow_pass(ratio<0.7)"] = {"n": len(lag), "median": round(float(np.median(lag)), 1) if lag else None}
    return out


def analyse(system, events):
    D = load(system, events)
    if D is None:
        return None
    tr = D["n_inc"] > 0
    ctrl = (~tr) & (~D["near_any"].astype(bool))
    days = sorted(set(D["day"]))
    res = {"system": system, "days": days, "queries": int(len(D["y"])), "treated_queries": int(tr.sum()),
           "near_miss_queries": int(D["near_any"].sum()),
           "incidents_on_a_path": int(len(set(D["inc"][tr]))),
           "incidents_active_on_these_days": None,
           "all": {"nowcast_mae": round(mae(D["b"] - D["y"]), 1), "v4_mae": round(mae(D["p"] - D["y"]), 1)}}
    city = {"tgsrtc": "hyderabad", "delhi": "delhi", "bmtc": "bengaluru", "mtc": "chennai"}[system]
    bounds = [day_bounds(d) for d in days]
    res["incidents_active_on_these_days"] = sum(1 for e in events if e["city"] == city
                                                and any(e["t_start"] < b1 and e["t_end"] > b0 for b0, b1 in bounds))
    if tr.sum() == 0:
        return res
    idx = np.flatnonzero(tr)
    cells = control_means(D, ctrl)
    late_c, late_bc, level = matched(D, idx, cells)
    ok_c = ~np.isnan(late_c)                     # a treated query with no control cell at any level is dropped
    idx, late_c, late_bc, level = idx[ok_c], late_c[ok_c], late_bc[ok_c], level[ok_c]
    late = D["y"][idx] - D["p"][idx]
    late_b = D["y"][idx] - D["b"][idx]
    excess = late - late_c
    excess_b = late_b - late_bc
    inc = D["inc"][idx]
    cl = collections.defaultdict(list)
    clb = collections.defaultdict(list)
    for e_, x, xb in zip(inc, excess, excess_b):
        cl[int(e_)].append(x); clb[int(e_)].append(xb)
    pen = D["pen"][idx]
    pe = (D["p"][idx] + pen) - D["y"][idx]
    t = {"n": int(len(idx)), "incidents": int(len(cl)),
         "control_cell_level_used": {"route_stop_hour": int((level == 0).sum()), "route_hour": int((level == 1).sum()),
                                     "hour_horizon": int((level == 2).sum())},
         "nowcast_mae": round(mae(D["b"][idx] - D["y"][idx]), 1),
         "v4_mae": round(mae(D["p"][idx] - D["y"][idx]), 1),
         "v4_plus_penalty_mae": round(mae(pe), 1),
         "v4_median_lateness_s": round(float(np.median(late)), 1),
         "control_median_lateness_s": round(float(np.nanmedian(late_c)), 1),
         "v4_excess_lateness_mean_s": round(float(np.mean(excess)), 1),
         "v4_excess_lateness_ci95": boot(cl),
         "nowcast_excess_lateness_mean_s": round(float(np.mean(excess_b)), 1),
         "nowcast_excess_lateness_ci95": boot(clb),
         "penalty_mean_s": round(float(np.mean(pen)), 1),
         "penalty_mae_change_s": round(mae(pe) - mae(D["p"][idx] - D["y"][idx]), 1),
         "penalty_mae_change_ci95": boot({k: [] for k in []})}
    # CI of the MAE change by incident
    dm = collections.defaultdict(list)
    for e_, a_, b_ in zip(inc, np.abs(pe), np.abs(D["p"][idx] - D["y"][idx])):
        dm[int(e_)].append(a_ - b_)
    t["penalty_mae_change_ci95"] = boot(dm)
    # by type / minutes since start / distance to the incident
    def split(name, keys):
        out = {}
        for key, sel in keys:
            if sel.sum() < 30:
                continue
            c2 = collections.defaultdict(list)
            for e_, x in zip(inc[sel], excess[sel]):
                c2[int(e_)].append(x)
            out[key] = {"n": int(sel.sum()), "incidents": len(c2), "excess_lateness_s": round(float(np.mean(excess[sel])), 1),
                        "ci95": boot(c2), "v4_mae": round(mae(D["p"][idx][sel] - D["y"][idx][sel]), 1),
                        "v4_plus_penalty_mae": round(mae(pe[sel]), 1), "penalty_mean_s": round(float(pen[sel].mean()), 1)}
        t[name] = out
    typ = D["typ"][idx]
    split("by_type", [(ty, typ == ty) for ty in sorted(set(typ))])
    age = D["age_min"][idx]
    split("by_minutes_since_start", [(f"{a}-{b}", (age >= a) & (age < b)) for a, b in ((0, 15), (15, 45), (45, 120), (120, 1e9))])
    di = D["d_inc"][idx]
    split("by_metres_to_incident", [(f"{a}-{b}", (di >= a) & (di < b)) for a, b in ((100, 1000), (1000, 3000), (3000, 8000), (8000, 30000))])
    # calibrated penalty, leave-one-day-out: delay per type = median excess lateness of that type on other days
    cal = np.zeros(len(idx))
    for d in days:
        te = D["day"][idx] == d
        for ty in set(typ[te]):
            other = (~te) & (typ == ty)
            if other.sum() >= 30:
                cal[te & (typ == ty)] = max(0.0, float(np.median(excess[other])))
    ce = (D["p"][idx] + cal) - D["y"][idx]
    t["calibrated_penalty"] = {"mean_s": round(float(cal.mean()), 1), "mae": round(mae(ce), 1),
                               "mae_change_s": round(mae(ce) - mae(D["p"][idx] - D["y"][idx]), 1)}
    dmc = collections.defaultdict(list)
    for e_, a_, b_ in zip(inc, np.abs(ce), np.abs(D["p"][idx] - D["y"][idx])):
        dmc[int(e_)].append(a_ - b_)
    t["calibrated_penalty"]["mae_change_ci95"] = boot(dmc)
    # residual model with incident features (route a), against the same model without them (ablation)
    if len(days) >= 3 and tr.sum() >= 1000:
        held = np.isin(D["day"], days)
        full = residual_model(D, tr, days, True)
        abl = residual_model(D, tr, days, False)
        e0 = np.abs(D["p"] - D["y"])
        ef, ea = np.abs(full - D["y"]), np.abs(abl - D["y"])
        t["residual_model_lodo"] = {
            "treated_mae_with_incident_features": round(float(ef[idx].mean()), 1),
            "treated_mae_with_incident_features(iw)": boot_iw(clusters(inc, ef[idx])),
            "treated_mae_without_incident_features(iw)": boot_iw(clusters(inc, ea[idx])),
            "treated_mae_without_incident_features": round(float(ea[idx].mean()), 1),
            "treated_v4_mae": round(float(e0[idx].mean()), 1),
            "incident_features_gain_on_treated_s(iw)": boot_iw(clusters(inc, ef[idx] - ea[idx])),
            "all_queries_mae_v4": round(float(e0[held].mean()), 1),
            "all_queries_mae_with_incident_features": round(float(ef[held].mean()), 1),
            "all_queries_mae_without_incident_features": round(float(ea[held].mean()), 1),
            "untreated_mae_change_with_incident_features_s": round(float((ef - e0)[held & ~tr].mean()), 2)}
    t["incident_weighted"] = {
        "v4_excess_lateness_s": boot_iw(cl), "nowcast_excess_lateness_s": boot_iw(clb),
        "penalty_mae_change_s": boot_iw(dm), "calibrated_penalty_mae_change_s": boot_iw(dmc),
        "nowcast_mae": boot_iw(clusters(inc, np.abs(D["b"][idx] - D["y"][idx]))),
        "v4_mae": boot_iw(clusters(inc, np.abs(D["p"][idx] - D["y"][idx]))),
        "v4_plus_calibrated_penalty_mae": boot_iw(clusters(inc, np.abs(ce))),
        "v4_plus_penalty_mae": boot_iw(clusters(inc, np.abs(pe)))}
    dur = np.array([(events[i]["t_end"] - events[i]["t_start"]) / 60 for i in inc])
    short = dur <= 360
    t["events_up_to_6h_only"] = {"n": int(short.sum()), "incidents": len(set(inc[short])),
                                 "v4_excess_lateness_s(iw)": boot_iw(clusters(inc[short], excess[short])),
                                 "v4_mae": round(mae(D["p"][idx][short] - D["y"][idx][short]), 1),
                                 "v4_plus_penalty_mae": round(mae(pe[short]), 1),
                                 "penalty_mae_change_s(iw)": boot_iw(clusters(inc[short], (np.abs(pe) - np.abs(D["p"][idx] - D["y"][idx]))[short]))}
    # the same spots in the hour BEFORE the alert's start and the hour AFTER its end (no alert out)
    for tag in ("pre", "post"):
        tm_ = (D[f"n_inc_{tag}"] > 0) & ~tr & ~D["near_any"].astype(bool)
        if tm_.sum() < 30:
            continue
        ii = np.flatnonzero(tm_)
        lc, _lbc, _lv = matched(D, ii, cells)
        ex = (D["y"][ii] - D["p"][ii]) - lc
        ok_ = ~np.isnan(ex)
        t[f"window_{tag}"] = {"n": int(ok_.sum()), "incidents": len(set(D[f"inc_{tag}"][ii][ok_])),
                             "v4_excess_lateness_s(iw)": boot_iw(clusters(D[f"inc_{tag}"][ii][ok_], ex[ok_]))}
    res["treated"] = t
    # the 80% band on treated queries: does it still cover?
    cov = (D["lo"][idx] <= D["y"][idx]) & (D["y"][idx] <= D["hi"][idx])
    covc = (D["lo"][ctrl] <= D["y"][ctrl]) & (D["y"][ctrl] <= D["hi"][ctrl])
    res["band80_coverage_pct"] = {"treated": round(100 * float(cov.mean()), 1), "controls": round(100 * float(covc.mean()), 1)}
    res["_excess"] = (inc, excess, age, typ, D["day"][idx])
    return res


def charts(results, gps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in results if r and r.get("treated") and r["treated"]["incidents"] >= 10]
    if ok:
        bars = [("nowcast", lambda t: t["incident_weighted"]["nowcast_mae"], "#9aa5b1"),
                ("v4 (live today)", lambda t: t["incident_weighted"]["v4_mae"], "#1f6feb"),
                ("v4 + incident_penalty.py priors", lambda t: t["incident_weighted"]["v4_plus_penalty_mae"], "#d1495b"),
                ("v4 + calibrated per-type delay", lambda t: t["incident_weighted"]["v4_plus_calibrated_penalty_mae"], "#edae49"),
                ("v4 + residual model, no incident features", lambda t: (t.get("residual_model_lodo") or {}).get("treated_mae_without_incident_features(iw)"), "#66a182"),
                ("v4 + residual model + incident features", lambda t: (t.get("residual_model_lodo") or {}).get("treated_mae_with_incident_features(iw)"), "#2e4057")]
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        x = np.arange(len(ok))
        w = 0.8 / len(bars)
        for j, (lab, f, col) in enumerate(bars):
            vals = [f(r["treated"]) for r in ok]
            if all(v is None for v in vals):
                continue
            m = [v[0] if v else np.nan for v in vals]
            err = [[v[0] - v[1][0] if v else 0 for v in vals], [v[1][1] - v[0] if v else 0 for v in vals]]
            ax.bar(x + (j - (len(bars) - 1) / 2) * w, m, w, yerr=err, capsize=2, label=lab, color=col)
        ax.set_xticks(x, [f"{r['system']}: {r['treated']['incidents']} incidents, {r['treated']['n']:,} predictions" for r in ok])
        ax.set_ylabel("mean absolute error (s), incident-weighted")
        ax.set_title("Predictions whose remaining path crosses an active police alert (95% CI over incidents)")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout(); fig.savefig(RESULTS / "mae_treated.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.2))
        wins = [("hour before the alert started", "window_pre"), ("while the alert was active", None),
                ("hour after it ended", "window_post")]
        for j, r in enumerate(ok):
            t = r["treated"]
            vals = [t["incident_weighted"]["v4_excess_lateness_s"] if k is None else (t.get(k) or {}).get("v4_excess_lateness_s(iw)")
                    for _lab, k in wins]
            xs = np.arange(3) + (j - (len(ok) - 1) / 2) * 0.25
            m = [v[0] if v else np.nan for v in vals]
            err = [[v[0] - v[1][0] if v else 0 for v in vals], [v[1][1] - v[0] if v else 0 for v in vals]]
            ax.errorbar(xs, m, yerr=err, fmt="o", capsize=4, label=r["system"])
        ax.set_xticks(np.arange(3), [w_[0] for w_ in wins])
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_ylabel("v4 excess lateness vs matched controls (s)\n(+ = bus later than v4 said)")
        ax.set_title("Same incident spots: before, during and after the police alert")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(RESULTS / "excess_lateness_windows.png", dpi=150); plt.close(fig)
    g = {s: v for s, v in gps.items() if v and v["n_events"] >= 10}   # 1-3 incidents make no curve
    if g:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for s, v in g.items():
            b = v["by_minutes_since_alert"]
            ks = list(b)
            xs = [np.mean([float(x) for x in k.split("..")]) for k in ks]
            ys = [b[k]["median_speed_ratio"] for k in ks]
            ln, = ax.plot(xs, ys, marker="o", label=f"{s}: alert day ({v['n_events']} incidents, {v['n_passes']:,} bus passes)")
            ax.fill_between(xs, [b[k]["ci95"][0] for k in ks], [b[k]["ci95"][1] for k in ks], color=ln.get_color(), alpha=0.15)
            pb = v.get("placebo_by_minutes") or {}
            if pb:
                pk = list(pb)
                ax.plot([np.mean([float(x) for x in k.split("..")]) for k in pk], [pb[k]["median_speed_ratio"] for k in pk],
                        ls=":", color=ln.get_color(), label=f"{s}: same spots & clock time, other days (placebo)")
        ax.axhline(1, color="#444", lw=0.8); ax.axvline(0, color="#999", lw=0.8, ls="--")
        ax.set_xlabel("minutes since the police alert's start time")
        ax.set_ylabel("bus speed at the spot /\nsame spot, same time, other days")
        ax.set_title("What the buses' own GPS saw at each incident (median, 95% CI)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(RESULTS / "gps_event_study.png", dpi=150); plt.close(fig)


def main():
    events = json.loads((WORK / "events.json").read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    out, rows = {}, []
    for system in ("tgsrtc", "delhi", "bmtc", "mtc"):
        r = analyse(system, events)
        if r is None:
            continue
        ex = r.pop("_excess", None)
        out[system] = r
        if ex is not None:
            for e_, x, a_, ty, d in zip(*ex):
                rows.append((system, int(e_), events[int(e_)]["id"], ty, d, round(float(a_), 1), round(float(x), 1)))
    gps = {s: gps_summary(s) for s in ("tgsrtc", "delhi", "bmtc", "mtc")}
    out["gps_event_study"] = gps
    # how many incidents a calibration needs: the per-incident spread of mean excess lateness
    per_inc = collections.defaultdict(list)
    for s, e_, _id, ty, _d, _a, x in rows:
        per_inc[(s, e_)].append(x)
    means = np.array([np.mean(v) for v in per_inc.values()]) if per_inc else np.zeros(0)
    if len(means) >= 5:
        sd = float(np.std(means, ddof=1))
        out["calibration_need"] = {"incidents_measured": int(len(means)), "sd_of_incident_mean_excess_s": round(sd, 1),
                                   "incidents_for_+-30s_per_type(95%)": int(np.ceil((1.96 * sd / 30) ** 2)),
                                   "incidents_for_+-60s_per_type(95%)": int(np.ceil((1.96 * sd / 60) ** 2))}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    import csv
    with open(RESULTS / "per_incident_excess.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "event", "id", "type", "day", "age_min", "excess_lateness_s"])
        agg = collections.defaultdict(list)
        for s, e_, iid, ty, d, a_, x in rows:
            agg[(s, e_, iid, ty, d)].append((a_, x))
        for (s, e_, iid, ty, d), v in sorted(agg.items()):
            w.writerow([s, e_, iid, ty, d, round(float(np.median([a for a, _ in v])), 1), round(float(np.mean([x for _, x in v])), 1)])
    charts(list(out[s] for s in ("tgsrtc", "delhi", "bmtc", "mtc") if s in out), gps)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
