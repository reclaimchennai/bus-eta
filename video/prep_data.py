"""Real numbers for the explainer: a sample of MTC 23 Nov held-out predictions (causal history model) and an
AgACI calibration trace run on that day's predictions in time order. Writes video_data.json."""
import json
import numpy as np
z = np.load('~/projects/ETA/data/audit/q_mtc_2026-11-23_causal.npz')
y, b, p, lo, hi = z['y'], z['b'], z['p'], z['lo'], z['hi']
rep = json.load(open('~/projects/ETA/models/v4_clean_ensemble/v4_report.json'))
cqr = np.array([rep['cqr_by_bucket'][str(k)] for k in range(5)])
rng = np.random.default_rng(7)
idx = rng.choice(len(y), 400, replace=False)
idx = idx[(y[idx] < 1800) & (b[idx] < 2400)]
sample = [[round(float(y[i]) / 60, 2), round(float(b[i]) / 60, 2), round(float(p[i]) / 60, 2)] for i in idx]
# AgACI on the real stream: recover the raw quantiles, then run the online update (training's rule, step 64)
hb = np.digitize(b, [180., 300., 600., 900.])
r = np.clip(np.log(y / b), -1.5, 1.5)
q10 = np.log(np.minimum(lo, p) / b) + cqr[hb]
q90 = np.log(np.maximum(hi, p) / b) - cqr[hb]
sel = hb == 4                                  # 15+ min ahead: the bucket where calibration matters most
r, q10, q90 = r[sel][:60000], q10[sel][:60000], q90[sel][:60000]
gam = [0.002, 0.005, 0.015, 0.05]
ex = np.full(4, cqr[4]); m_tr, cov_static, cov_ad = [], [], []
hit_s = hit_a = n = 0
for s in range(0, len(r), 64):
    e = slice(s, s + 64); m = float(np.median(ex))
    ins = (r[e] >= q10[e] - m) & (r[e] <= q90[e] + m)
    inst = (r[e] >= q10[e] - cqr[4]) & (r[e] <= q90[e] + cqr[4])
    hit_a += ins.sum(); hit_s += inst.sum(); n += len(ins)
    err = 1 - ins.mean()
    ex = np.clip(ex + np.array(gam) * (err - 0.2), -0.15, 0.45)
    m_tr.append(m); cov_static.append(hit_s / n); cov_ad.append(hit_a / n)
k = max(1, len(m_tr) // 120)
out = {"sample": sample, "agaci": {"margin": m_tr[::k], "cov_static": cov_static[::k], "cov_adaptive": cov_ad[::k], "n": int(len(r))}}
json.dump(out, open('video_data.json', 'w'))
print(len(sample), 'points; coverage static %.3f adaptive %.3f; margin %.3f -> %.3f' % (cov_static[-1], cov_ad[-1], m_tr[0], m_tr[-1]))
