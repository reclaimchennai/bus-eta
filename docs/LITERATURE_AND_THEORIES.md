# Citation-Verified Literature & Theory Compendium for Indian Multi-City Bus ETA

> Compiled: 2026-11-24 | All DOIs, arXiv IDs, authors, and venues verified via web search.  
> Complete BibTeX entries in [`REFERENCES.bib`](./REFERENCES.bib).

---

## 1. Indian Bus Transit, Heterogeneous Traffic & Monsoon Studies

### 1.1 IIT Madras — Chennai MTC, Kalman Filtering & Heterogeneous Traffic
- **Vanajakshi, Subramanian & Sivanandan (2009)** — *IET Intelligent Transport Systems* (`10.1049/iet-its:20080013`): Real-time Chennai MTC bus travel time prediction via recursive **Kalman Filtering** ($x(k) = a(k-1)x(k-1) + w(k-1)$) using bus GPS probes under non-lane-based mixed traffic.
- **Padmanaban, Vanajakshi & Subramanian (2009)** — *IEEE IVS* (`10.1109/IVS.2009.5164409`): Proves strict **Running Time + Dwell Time decomposition** ($T_{\text{total}} = \sum T_{\text{run}, i} + \sum T_{\text{dwell}, j}$) outperforms lumped models on Chennai MTC because dwell is driven by passenger demand while running time is driven by ambient traffic.
- **Kumar & Vanajakshi (2015)** — *European Transport Research Review* (`10.1007/s12544-015-0170-8`): Seasonal ARIMA captures strong Indian arterial diurnal/weekly periodicity even with only days of history (cold-start prior).
- **Kumar, Vanajakshi & Subramanian (2017)** — *Transportation Research Part C* (`10.1016/j.trc.2017.04.002`): **Time-space discretization** (uniform 100–500 m spatial micro-segments + time bins) fuses historical pattern matching with real-time Kalman updates across non-uniform GPS intervals.
- **Achar, Bharathi, Kumar & Vanajakshi (2020)** — *IEEE TITS* (`10.1109/TITS.2019.2909314`): **Spatial Kalman Filter** ($x_s = A_s x_{s-1} + b_s + w_s$) indexed by space rather than time, exploiting upstream-to-downstream spatial correlation along Chennai corridors to curb multi-stop ETA drift.

### 1.2 IIIT Delhi — Delhi DTC / Chartr, Generative & Stochastic Models
- **Charul & Biyani (2019)** — `arXiv:1906.09925` (*local PDF: `1906.09925v1.pdf`*): Masked PixelCNN autoregressive generative model over 2D `H × K` trip×stop matrices on Delhi DTC buses. Captures joint stop-to-stop distributions without external traffic APIs, but requires per-route models and dense multi-month history.
- **Jain, Kumar, Balodi & Biyani (2020)** — *IEEE ITSC* (`10.1109/ITSC45102.2020.9294589`, `arXiv:1910.08903`): Large-scale Delhi 500+ route GPS dataset; exposes severe headway irregularity and highlights mandatory GPS-to-trip segmentation & deadheading hygiene.
- **Altaf & Biyani (2023)** — `arXiv:2306.15941`: Gaussian Process spatiotemporal variance modeling for stochastic trip planning on Delhi transit.

### 1.3 IISc Bengaluru, IIT Bombay, IIT Roorkee & Monsoon Elasticity
- **Deeshma & Verma (2015)** — *Transportation Letters* (`10.1179/1942787514Y.0000000032`): IISc Bengaluru BMTC travel-time modeling quantifying signalized-junction bottlenecks and reliability indices (TTI, BTI, PTI).
- **Kathuria, Parida, Sekhar & Pathak (2016, 2020)** — *J. Public Transportation* (`10.5038/2375-0901.19.2.10`) & *J. Transp. Eng. Part A* (`10.1061/JTEPBS.0000357`): Dissects Indian bus stop lost time into deceleration + door clearance + dwell + re-entry merge delay into mixed traffic (50–80 m stop-influence zone).
- **Soni & Chandel (2020)** — *Transportation Research Procedia* (`10.1016/j.trpro.2020.08.269`): IIT Bombay Greater Mumbai study finding monsoon rainfall increases arterial travel times by **8% to 138%**, driven heavily by waterlogging corridors.
- **Banik & Vanajakshi (2024)** — *Transportation in Developing Economies* (`10.1007/s40890-024-00219-9`): IIT Madras Chennai study proving rainfall degrades mobility **non-linearly** (max TTI spikes up to 457%, tail quantiles degrade far faster than the median) with strong `rain × peak_hour` interactions.

---

## 2. Foundational Transit & Traffic Flow Theories

1. **Bus Bunching & Headway Instability** — **Newell & Potts (1964)** (*ARRB*); **Daganzo (2009)** (*TRB*, `10.1016/j.trb.2009.04.002`); **Bartholdi & Eisenstein (2012)** (*TRB*, `10.1016/j.trb.2011.11.001`):
   $$T_{\text{dwell}, s} = a_s + b \cdot \lambda_s \cdot h_{\text{lead}, s}$$
   Uncontrolled headways diverge geometrically along a route; delayed leaders accumulate crowds while bunched trailers skip stops. **Applied via:** live `lead_headway_s`, `any_headway_s`, and bunching regime features.
2. **Kinematic Waves & Shockwave Asymmetry** — **Lighthill & Whitham (1955)** (`10.1098/rspa.1955.0089`); **Richards (1956)** (`10.1287/opre.4.1.42`); **Daganzo (1994)** (*CTM*, `10.1016/0191-2615(94)90002-7`):
   Congested shockwaves propagate **upstream** ($w \approx -15\text{ to }-20\text{ km/h}$), whereas free-flow platoons move **downstream**. **Applied via:** separate `up_live_v_15m` and `dn_live_v_15m` features.
3. **Heterogeneous / Weak-Lane-Discipline Traffic Theory** — **Mallikarjuna & Rao (2006)** (`10.1080/18128600608685661`); **George, Vanajakshi & Subramanian (2020)** (*IEEE Access*, `10.1109/ACCESS.2019.2963273`):
   Area Occupancy (AO) and **speed dispersion ($\text{CV}_v = \sigma_v / \mu_v$)** capture lateral two-wheeler seepage & roadside friction where large buses slow disproportionately. **Applied via:** `corr_speed_cv_30m` and `corr_stopgo_30m`.
4. **Dwell Time, Crowding & Two-Stage Hurdle Models** — **TCQSM (TRB 2013)** (`10.17226/24766`); **Tirachini (2013)** (`10.1080/18128602.2010.520277`); **Sun et al. (2014)** (`10.1016/j.tra.2014.09.007`); **Rashidi, Ranjitkar & Hadas (2014, 2022)** (`10.3141/2418-09`, `10.1080/01441647.2021.2023692`); **Feng (2021)** (`10.1186/s40488-021-00121-4`):
   Zero-inflated skip-stop behavior requires a **two-stage Hurdle model**: $E[T_{\text{dwell}} \mid X] = P(T_{\text{dwell}}>0 \mid X) \cdot E[T_{\text{dwell}} \mid T_{\text{dwell}}>0, X]$. **Applied via:** `src/train_dwell_model.py`.
5. **Driver Behavior & Kinematic Driving Style** — **Cats (2019)** (`10.1061/JTEPBS.0000201`); **Yin & Zhang (2023, 2024)** (`10.3390/fi15070222`, `10.3390/app14072935`):
   Online driver pace ratio (`veh_pace_ewma`) and historical vehicle pace (`veh_hist_pace`) separate driver aggressiveness from ambient congestion without needing driver IDs.

---

## 3. Contemporary ML, Deep Learning, Conformal & Transfer ETA Literature (2018–2026)

### 3.1 Industry & SOTA ETA Architectures
- **Google Maps / DeepMind Supersegment GNN** — Derrow-Pinion et al. (2021), *CIKM '21* (`10.1145/3459637.3481916`, `arXiv:2108.11482`): Supersegment graph networks + composite Huber/$L_1$/NLL multi-horizon loss.
- **Uber DeeprETA** — Hu et al. (2022), `arXiv:2206.02127`: Residual post-processing over deterministic baseline ($\widehat{\text{ETA}} = \text{base} + \hat{R}$) + **Asymmetric Huber Loss** (penalizing early-arriving/missed-bus errors more heavily).
- **Baidu ConSTGAT** — Fang et al. (2020), *KDD '20* (`10.1145/3394486.3403320`): 3D spatiotemporal graph attention + contextual junction/turn modeling.
- **DiDi WDR & HetETA** — Wang et al. (2018, *KDD '18*, `10.1145/3219819.3219900`); Hong et al. (2020, *KDD '20*, `10.1145/3394486.3403294`): Wide-Deep-Recurrent + Heterogeneous Information Network with **tri-periodic lags** (recent, daily, weekly).
- **DeepTTE** — Wang et al. (2018), *AAAI '18* (`10.1609/aaai.v32i1.11877`): Geo-Conv + multi-task local-segment + full-path supervision.

### 3.2 Bus-Specific Deep & Hybrid Models (2022–2026)
- **MAGNN** — Ma et al. (2022), *Expert Systems with Applications* (`10.1016/j.eswa.2022.117057`): Multi-attention GNN for sparse bus routes borrowing live traffic from overlapping high-frequency routes.
- **ArrivalNet** — Li, Wolf & Wang (2024), `arXiv:2410.14742`: 2D intra-day × inter-day temporal variation modeling.
- **PMLNet** — Lei et al. (2025), *Applied Sciences* (`10.3390/app15148104`): Multi-route shared-segment partition-and-combination + real-time local calibration factor.
- **Multi-Task Graph Transformer** — Yusuf, Rasheed & Lindseth (2026), `arXiv:2606.00572`: Jointly decouples running time, dwell time, and crowding via MMoE Graph Transformer.

### 3.3 Probabilistic ETA & Conformal Uncertainty under Distribution Shift
- **CQR** — Romano, Patterson & Candès (2019), *NeurIPS 2019* (`arXiv:1905.03222`): Distribution-free, input-adaptive finite-sample coverage via conformalized quantile regression.
- **ACI** — Gibbs & Candès (2021), *NeurIPS 2021* (`arXiv:2106.00170`): Online adaptive conformal inference ($\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$) under non-stationary distribution shift (monsoon/disruptions).
- **AgACI** — Zaffran et al. (2022), *ICML 2022* (`arXiv:2202.07282`): Parameter-free aggregated ACI across multiple $\gamma$ experts.
- **NGBoost** — Duan et al. (2020), *ICML 2020* (`arXiv:1910.03225`): Natural Gradient Boosting for full parametric LogNormal/Gamma distributions.
- **UncertBAT** — Song, Li, Chung & Ye (2026), *Computer-Aided Civil and Infrastructure Engineering* (`10.1016/j.cacaie.2026.100006`): Horizon- and regime-grouped (Mondrian) CQR specifically for real-time Bus Arrival Time uncertainty.

### 3.4 Cross-City Transfer & Foundation Models (2019–2026)
- **MetaST** — Yao et al. (2019), *WWW '19* (`arXiv:1901.08518`); **RegionTrans** — Wang et al. (2019), *IJCAI-19* (`10.24963/ijcai.2019/262`); **CrossTReS** — Jin, Chen & Yang (2022), *KDD '22* (`10.1145/3534678.3539250`); **ST-GFSL** — Lu et al. (2022), *KDD '22* (`10.1145/3534678.3539281`): Selective source-city re-weighting and city-agnostic normalized features to avoid negative transfer into cold-start cities (Gurugram NCR).
- **Chronos / Chronos-2** — Ansari et al. (2024, 2025, `arXiv:2403.07815`, `arXiv:2510.15821`); **TimesFM** — Das et al. (2024, *ICML*, `arXiv:2310.10688`); **Moirai** — Woo et al. (2024, *ICML*, `arXiv:2402.02592`): Zero-shot multivariate corridor speed priors for brand-new cities.
- **UrbanGPT** — Li et al. (2024, *KDD '24*, `10.1145/3637528.3671578`); **ST-LLM** — Liu et al. (2024, `arXiv:2401.10134`); **LLM Traffic Incident Impact** (2025, *IEEE ITSC*, `arXiv:2507.04803`): Async LLM extraction of unstructured traffic police / monsoon / festival advisories into structured corridor slowdown multipliers (`src/llm_disruption_sidecar.py`).
