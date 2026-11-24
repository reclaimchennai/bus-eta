"""How the bus maps' v4 arrival-time estimator works: a short manim explainer.

Render:  .venv/bin/manim -qh eta_explainer.py EtaExplainer      (1080p30, ~3 min)
No LaTeX needed: formulas are Pango markup (Unicode + <sub>/<sup>).
"""
from manim import *

BG = "#14161a"
BLUE_ = "#58C4DD"
YEL = "#FFD25A"
TEAL = "#5CD0B3"
RED_ = "#FC6255"
GREY_ = "#8A8F98"
INK = "#ECECEC"
config.background_color = BG


def M(s, size=34, color=INK, **kw):
    """A formula line: Pango markup, serif italic look for variables."""
    return MarkupText(s, font="Times New Roman", font_size=size, color=color, **kw)


def T(s, size=30, color=INK, **kw):
    return Text(s, font="Helvetica Neue", font_size=size, color=color, **kw)


def footer(*lines):
    g = VGroup(*[Text(l, font="Helvetica Neue", font_size=13, color=GREY_) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
    rule = Line(LEFT * 6.9, RIGHT * 6.9, stroke_width=1, color="#3a3e46")
    box = VGroup(rule, g).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    return box.to_edge(DOWN, buff=0.18).to_edge(LEFT, buff=0.3)


class EtaExplainer(Scene):
    def clear_all(self, run_time=0.6):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=run_time)

    def construct(self):
        self.title()
        self.problem()
        self.nowcast()
        self.residual()
        self.features()
        self.ensemble()
        self.losses()
        self.interval()
        self.honesty()
        self.results()
        self.outro()

    # 1 ─────────────────────────────────────────────
    def title(self):
        t = T("When will my bus come?", 54, YEL)
        s = T("the v4 arrival-time estimator behind the live bus maps", 28, GREY_)
        VGroup(t, s).arrange(DOWN, buff=0.35)
        self.play(Write(t), run_time=1.4)
        self.play(FadeIn(s, shift=UP * 0.2))
        self.wait(1.2)
        self.clear_all()

    # 2 ─────────────────────────────────────────────
    def problem(self):
        pts = [LEFT * 6 + DOWN * 0.6, LEFT * 3.6 + UP * 0.9, LEFT * 1 + UP * 0.2, RIGHT * 1.6 + UP * 1.3, RIGHT * 4 + DOWN * 0.2, RIGHT * 6 + UP * 0.6]
        route = VMobject(color=BLUE_, stroke_width=6).set_points_smoothly(pts)
        alphas = [0.0, 0.18, 0.36, 0.55, 0.74, 0.9, 1.0]
        stops = VGroup(*[Dot(route.point_from_proportion(a), radius=0.09, color=INK) for a in alphas])
        labels = VGroup(*[T(f"stop {i}", 18, GREY_).next_to(stops[i], DOWN, buff=0.15) for i in range(len(alphas))])
        self.play(Create(route), run_time=1.5)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in stops], lag_ratio=0.1), FadeIn(labels))
        bus = RoundedRectangle(width=0.5, height=0.26, corner_radius=0.06, fill_color=YEL, fill_opacity=1, stroke_width=0)
        bus.move_to(route.point_from_proportion(0.27))
        fix = T("last GPS fix  t₀", 22, YEL).next_to(bus, UP, buff=0.25)
        self.play(FadeIn(bus, scale=0.5), Write(fix))
        target = stops[4]
        ring = Circle(radius=0.22, color=TEAL, stroke_width=4).move_to(target)
        q = M("arrival at stop k:   <i>t</i><sub>k</sub> = <i>t</i><sub>0</sub> + <i>y</i><sub>k</sub>   ?", 34, TEAL).to_edge(UP, buff=0.6)
        self.play(Create(ring), Write(q))
        self.wait(0.8)
        note = T("GPS fixes arrive every 10 s (Delhi) to ~3 min (Chennai), late and noisy", 22, GREY_).next_to(q, DOWN, buff=0.25)
        self.play(FadeIn(note))
        self.play(MoveAlongPath(bus, Line(bus.get_center(), route.point_from_proportion(0.35))), run_time=1.2, rate_func=smooth)
        self.wait(1)
        self.route, self.stops, self.bus = route, stops, bus
        self.clear_all()

    # 3 ─────────────────────────────────────────────
    def nowcast(self):
        h = T("Step 1 · the physics baseline (nowcast)", 34, YEL).to_edge(UP, buff=0.5)
        self.play(Write(h))
        segs = VGroup()
        xs = [-5.5, -3.3, -1.4, 0.8, 3.0, 5.4]
        cols = [TEAL, YEL, RED_, YEL, TEAL]
        for i in range(5):
            segs.add(Line([xs[i], 1.2, 0], [xs[i + 1], 1.2, 0], stroke_width=10, color=cols[i]))
        dots = VGroup(*[Dot([x, 1.2, 0], radius=0.08, color=INK) for x in xs])
        vlab = VGroup(*[M(f"<i>v</i><sub>{i + 1}</sub>", 26, cols[i]).next_to(segs[i], UP, buff=0.15) for i in range(5)])
        self.play(Create(segs), FadeIn(dots), Write(vlab))
        f = M("ŷ<sub>k</sub><sup>nc</sup> = Σ<sub><i>i</i> ≤ <i>k</i></sub>  <i>d</i><sub>i</sub> / ( <i>v</i><sub>i</sub> · <i>p</i>(Δt) )", 40).shift(DOWN * 0.1)
        self.play(Write(f), run_time=1.5)
        expl = VGroup(
            M("<i>v</i><sub>i</sub> :  pooled metres ÷ seconds on segment <i>i</i> — buses in the last 15 min,", 24, GREY_),
            M("       else its usual speed this hour (7-day history), else the city's", 24, GREY_),
            M("<i>p</i>(Δt) = 1 + (pace − 1)·e<sup>−Δt/170 s</sup> :  this bus's own pace, fading with time ahead", 24, GREY_),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(f, DOWN, buff=0.45)
        self.play(FadeIn(expl, lag_ratio=0.2), run_time=1.6)
        self.play(FadeIn(footer("Every bus is a traffic probe for the others: segment speeds pooled across routes (Σ distance ÷ Σ time, stops included).",
                                "Kinematic-wave background: Lighthill & Whitham (1955); Richards (1956).")))
        self.wait(2.2)
        self.clear_all()

    # 4 ─────────────────────────────────────────────
    def residual(self):
        h = T("Step 2 · learn what the physics gets wrong", 34, YEL).to_edge(UP, buff=0.5)
        self.play(Write(h))
        f1 = M("<i>r</i> = log( <i>y</i> / ŷ<sup>nc</sup> ),   clipped to [−1.5, 1.5]", 40, TEAL)
        f2 = M("ŷ = ŷ<sup>nc</sup> · exp( <i>r̂</i>(<b>x</b>) )", 44, YEL)
        VGroup(f1, f2).arrange(DOWN, buff=0.7)
        self.play(Write(f1), run_time=1.4)
        self.wait(0.5)
        self.play(TransformFromCopy(f1, f2), run_time=1.3)
        why = VGroup(
            T("• a ratio, not seconds: one model works for 1 min and 30 min ahead", 24, GREY_),
            T("• the model only has to learn corrections, so it stays small and stable", 24, GREY_),
            T("• if the model is unsure, r̂ ≈ 0 and we fall back to the physics", 24, GREY_),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(f2, DOWN, buff=0.6)
        self.play(FadeIn(why, lag_ratio=0.25), run_time=1.5)
        self.play(FadeIn(footer("Residual learning on top of a routing engine's estimate: Hu et al., DeeprETA (Uber), arXiv:2206.02127, 2022.",
                                "Wang et al., Learning to Estimate the Travel Time (WDR, DiDi), KDD 2018.")))
        self.wait(2.2)
        self.clear_all()

    # 5 ─────────────────────────────────────────────
    def features(self):
        h = T("Step 3 · the 74 things it looks at", 34, YEL).to_edge(UP, buff=0.45)
        self.play(Write(h))
        groups = [
            ("Baselines", TEAL, ["nowcast ŷ_nc", "7-day history ŷ_hist", "live corridor ŷ_corr", "live ÷ history ratio"]),
            ("Live traffic", BLUE_, ["corridor speed 5/15/30 min", "speed variability, stop-go share", "segments ahead & behind", "city-wide speed"]),
            ("This bus", YEL, ["pace vs model (EWMA)", "speed between last fixes", "time & km into trip", "bus type"]),
            ("Road ahead", RED_, ["distance, stops ahead", "sinuosity, turning angle", "OSM signals, flyovers", "rail crossings, roundabouts"]),
            ("When", "#C792EA", ["time of day (sin, cos)", "weekday, weekend", "holidays, festivals", "rain 1/3/6/24 h, temperature"]),
            ("Spacing", GREY_, ["headway to bus ahead", "on segment & corridor", "position on route", "route length"]),
        ]
        cards = VGroup()
        for name, col, items in groups:
            head = T(name, 24, col)
            body = VGroup(*[T(it, 17, INK) for it in items]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            card = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            box = SurroundingRectangle(card, buff=0.18, color=col, stroke_width=2, corner_radius=0.12)
            cards.add(VGroup(box, card))
        cards.arrange_in_grid(rows=2, cols=3, buff=(0.35, 0.35)).scale_to_fit_width(12.6).next_to(h, DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.15), run_time=2.4)
        self.play(FadeIn(footer("Path geometry & context: Fang et al., ConSTGAT (Baidu Maps), KDD 2020.  Bus-specific pace & dwell: Cats (2019); Yin & Zhang (2023).",
                                "Every feature uses only what was known at the moment of the prediction (strictly time-ordered replay).")))
        self.wait(3)
        self.clear_all()

    # 6 ─────────────────────────────────────────────
    def ensemble(self):
        h = T("Step 4 · three tree ensembles vote", 34, YEL).to_edge(UP, buff=0.5)
        self.play(Write(h))
        x = M("<b>x</b>", 44).shift(LEFT * 5.2)
        names = [("LightGBM", "leaf-wise trees", BLUE_, "0.40"), ("XGBoost", "depth-wise trees", TEAL, "0.50"), ("CatBoost", "symmetric trees", RED_, "0.10")]
        boxes = VGroup()
        for n, sub, c, w in names:
            b = RoundedRectangle(width=3.0, height=1.0, corner_radius=0.15, color=c, stroke_width=3)
            lab = VGroup(T(n, 26, c), T(sub, 16, GREY_)).arrange(DOWN, buff=0.06).move_to(b)
            boxes.add(VGroup(b, lab))
        boxes.arrange(DOWN, buff=0.35).shift(LEFT * 1.2)
        arrows_in = VGroup(*[Arrow(x.get_right(), b.get_left(), buff=0.15, stroke_width=3, color=GREY_) for b in boxes])
        out = M("<i>r̂</i>", 44, YEL).shift(RIGHT * 3.6)
        ws = VGroup(*[T(w, 22, c) for _, _, c, w in names])
        arrows_out = VGroup()
        for b, wt in zip(boxes, ws):
            a = Arrow(b.get_right(), out.get_left(), buff=0.15, stroke_width=3, color=GREY_)
            arrows_out.add(a)
            wt.move_to(a.point_from_proportion(0.45) + UP * 0.22)
        self.play(Write(x))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows_in], lag_ratio=0.1), LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.15))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows_out], lag_ratio=0.1), FadeIn(ws), Write(out))
        f = M("<i>r̂</i> = 0.40 <i>f</i><sub>LGB</sub>(<b>x</b>) + 0.50 <i>f</i><sub>XGB</sub>(<b>x</b>) + 0.10 <i>f</i><sub>Cat</sub>(<b>x</b>)", 32, YEL).to_edge(DOWN, buff=1.5)
        self.play(Write(f))
        self.play(FadeIn(footer("Ke et al., LightGBM, NeurIPS 2017.  Chen & Guestrin, XGBoost, KDD 2016.  Prokhorenkova et al., CatBoost, NeurIPS 2018.",
                                "Blend weights chosen on a held-out validation day; models trained on 5 systems: Chennai MTC, Bengaluru BMTC, Delhi, Mumbai, Gurugram.")))
        self.wait(2.5)
        self.clear_all()

    # 7 ─────────────────────────────────────────────
    def losses(self):
        h = T("Robust losses: a few crazy trips can't pull the average", 32, YEL).to_edge(UP, buff=0.5)
        self.play(Write(h))
        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 3, 1], x_length=6.2, y_length=3.6, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2}).shift(LEFT * 2.8 + DOWN * 0.3)
        d = 1.0
        sq = ax.plot(lambda e: 0.5 * e * e, x_range=[-2.45, 2.45], color=GREY_, stroke_width=3)
        hub = ax.plot(lambda e: 0.5 * e * e if abs(e) <= d else d * (abs(e) - 0.5 * d), x_range=[-3, 3], color=BLUE_, stroke_width=5)
        ph = ax.plot(lambda e: d * d * ((1 + (e / d) ** 2) ** 0.5 - 1), x_range=[-3, 3], color=TEAL, stroke_width=4)
        mae = ax.plot(lambda e: abs(e) * 0.75, x_range=[-3, 3], color=RED_, stroke_width=3)
        self.play(Create(ax), Create(sq), run_time=1)
        self.play(Create(hub), Create(ph), Create(mae), run_time=1.6)
        lg = VGroup(
            M("squared error  ½<i>e</i>²  (not used)", 22, GREY_),
            M("Huber:  ½<i>e</i>² if |<i>e</i>| ≤ δ,  else δ(|<i>e</i>| − ½δ)", 22, BLUE_),
            M("Pseudo-Huber:  δ²( √(1 + (<i>e</i>/δ)²) − 1 )", 22, TEAL),
            M("absolute error  |<i>e</i>|", 22, RED_),
            M("δ = 0.25 on <i>e</i> = <i>r</i> − <i>r̂</i>  (≈ ±28 % of the time)", 22, INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(RIGHT * 3.4 + DOWN * 0.2)
        self.play(FadeIn(lg, lag_ratio=0.2), run_time=1.6)
        self.play(FadeIn(footer("Huber, Robust Estimation of a Location Parameter, Annals of Mathematical Statistics, 1964.",
                                "LightGBM → Huber, XGBoost → Pseudo-Huber, CatBoost → MAE: three different error shapes, so their mistakes differ.")))
        self.wait(2.5)
        self.clear_all()

    # 8 ─────────────────────────────────────────────
    def interval(self):
        h = T("Step 5 · an honest range: “8 in 10 buses arrive within …”", 32, YEL).to_edge(UP, buff=0.5)
        self.play(Write(h))
        nl = NumberLine(x_range=[0, 14, 2], length=10, include_numbers=False, color=GREY_).shift(UP * 0.9)
        nums = VGroup(*[T(str(v), 20, GREY_).next_to(nl.n2p(v), DOWN, buff=0.15) for v in range(0, 15, 2)])
        unit = T("minutes", 18, GREY_).next_to(nl, RIGHT)
        self.play(Create(nl), FadeIn(nums), FadeIn(unit))
        band = Rectangle(width=nl.n2p(11)[0] - nl.n2p(6)[0], height=0.36, fill_color=TEAL, fill_opacity=0.35, stroke_width=0)
        band.move_to((nl.n2p(6) + nl.n2p(11)) / 2 + UP * 0.45)
        p50 = Triangle(color=YEL, fill_opacity=1).scale(0.12).rotate(PI).next_to(nl.n2p(8), UP, buff=0.32)
        self.play(GrowFromCenter(band), FadeIn(p50))
        lab = T("6–11 min", 22, TEAL).next_to(band, UP, buff=0.12)
        self.play(Write(lab))
        f1 = M("[ ŷ<sup>nc</sup> e<sup><i>q</i><sub>10</sub>(<b>x</b>) − <i>m</i><sub>b</sub></sup> ,  ŷ<sup>nc</sup> e<sup><i>q</i><sub>90</sub>(<b>x</b>) + <i>m</i><sub>b</sub></sup> ]", 34).shift(DOWN * 0.45)
        f2 = M("<i>m</i><sub>b</sub> ← <i>m</i><sub>b</sub> + γ ( 𝟙[arrival outside band] − 0.20 )", 32, TEAL).next_to(f1, DOWN, buff=0.35)
        f3 = M("four step sizes γ ∈ {0.002, 0.005, 0.015, 0.05}; the margin used is their median", 24, GREY_).next_to(f2, DOWN, buff=0.25)
        f4 = M("<i>q</i><sub>10</sub>, <i>q</i><sub>90</sub>: LightGBM quantile models;  <i>b</i>: horizon bucket (&lt;3, 3–5, 5–10, 10–15, 15+ min)", 22, GREY_).next_to(f3, DOWN, buff=0.18)
        self.play(Write(f1), run_time=1.4)
        self.play(Write(f2), run_time=1.2)
        self.play(FadeIn(f3), FadeIn(f4))
        self.play(FadeIn(footer("Romano, Patterson & Candès, Conformalized Quantile Regression, NeurIPS 2019.  Gibbs & Candès, Adaptive Conformal Inference, NeurIPS 2021.",
                                "Zaffran et al., Adaptive Conformal Predictions for Time Series (AgACI), ICML 2022.  Live: the margins adapt from the arrivals the service sees.")))
        self.wait(3)
        self.clear_all()

    # 9 ─────────────────────────────────────────────
    def honesty(self):
        h = T("Checking it honestly", 34, YEL).to_edge(UP, buff=0.5)
        self.play(Write(h))
        days = VGroup()
        for i, (lab, c) in enumerate([("train", BLUE_)] * 5 + [("val", TEAL), ("test", YEL)]):
            r = RoundedRectangle(width=1.4, height=0.7, corner_radius=0.1, color=c, fill_opacity=0.15)
            days.add(VGroup(r, T(lab, 20, c).move_to(r)))
        days.arrange(RIGHT, buff=0.15).shift(UP * 1.2)
        self.play(LaggedStart(*[FadeIn(d) for d in days], lag_ratio=0.08))
        pts = VGroup(
            T("• whole days held out; the test day is never seen in training", 24),
            T("• the fleet is replayed second by second: a prediction sees only the past", 24),
            T("• history speeds rebuilt from the 7 days BEFORE the test day", 24),
            T("   (the first published scores had used a model that contained it)", 22, GREY_),
            T("• then a live shadow run against the buses' real arrivals", 24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(days, DOWN, buff=0.6)
        self.play(FadeIn(pts, lag_ratio=0.2), run_time=2.2)
        self.wait(2.5)
        self.clear_all()

    # 10 ────────────────────────────────────────────
    def results(self):
        h = T("Live, 24 Nov 2026 evening: mean error, stops 1–30 min ahead", 30, YEL).to_edge(UP, buff=0.45)
        self.play(Write(h))
        data = [("Chennai MTC", 198, 158, 104, "MTC's own ETA"), ("Delhi", 199, 190, 126, "nowcast"), ("Bengaluru BMTC", 351, 94, 84, "BMTC's own ETA")]
        ax = Axes(x_range=[0, 3, 1], y_range=[0, 360, 60], x_length=9.5, y_length=4.2, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2}).shift(DOWN * 0.35 + RIGHT * 0.3)
        yn = VGroup(*[T(str(v), 16, GREY_).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in range(0, 361, 60)])
        ylab = T("seconds", 18, GREY_).rotate(PI / 2).next_to(yn, LEFT, buff=0.15)
        self.play(Create(ax), FadeIn(yn), FadeIn(ylab))
        bars = VGroup()
        for i, (city, shown, nc, v4, what) in enumerate(data):
            x0 = i + 0.5
            for j, (val, col) in enumerate([(shown, GREY_), (nc, BLUE_), (v4, YEL)]):
                x = x0 + (j - 1) * 0.26
                b = Rectangle(width=0.22, height=ax.y_axis.unit_size * val, fill_color=col, fill_opacity=0.9, stroke_width=0)
                b.move_to(ax.c2p(x, 0), aligned_edge=DOWN)
                n = T(str(val), 16, col).next_to(b, UP, buff=0.05)
                bars.add(VGroup(b, n))
            name = T(city, 20).next_to(ax.c2p(x0, 0), DOWN, buff=0.2)
            bars.add(name)
        leg = VGroup(
            VGroup(Square(0.18, fill_color=GREY_, fill_opacity=1, stroke_width=0), T("what the map showed before", 18, GREY_)).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.18, fill_color=BLUE_, fill_opacity=1, stroke_width=0), T("nowcast", 18, BLUE_)).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.18, fill_color=YEL, fill_opacity=1, stroke_width=0), T("v4", 18, YEL)).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.5).next_to(ax, UP, buff=0.12).align_to(ax, RIGHT)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in bars], lag_ratio=0.05), FadeIn(leg), run_time=2.4)
        self.play(FadeIn(footer("Same buses, same moments, same stops for all three. Held-out days offline: MTC −32 %, BMTC −35 %, Delhi −35 % vs nowcast;",
                                "Mumbai only −6 % once its history model is rebuilt properly. 80 % band covered 77–84 % of real arrivals.")))
        self.wait(3.5)
        self.clear_all()

    def outro(self):
        t = T("Every live bus, re-timed every 10 seconds", 40, YEL)
        s = T("Chennai · Bengaluru · Delhi · Mumbai · Gurugram  —  more systems next", 26, GREY_)
        VGroup(t, s).arrange(DOWN, buff=0.35)
        self.play(Write(t), run_time=1.2)
        self.play(FadeIn(s, shift=UP * 0.2))
        self.wait(2)
        self.play(FadeOut(t), FadeOut(s))
