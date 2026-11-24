"""How the bus maps' v4 arrival-time estimator works: a manim explainer (3Blue1Brown style).

Render:  .venv/bin/manim -qh eta_explainer.py EtaExplainer
Needs LaTeX (TeX Live) for the equations; real numbers come from video_data.json (prep_data.py).
"""
import json
import math
from pathlib import Path

from manim import *

BG = "#14161a"
BLUE_ = "#58C4DD"
YEL = "#FFD25A"
TEAL = "#5CD0B3"
RED_ = "#FC6255"
PURP = "#C792EA"
GREY_ = "#8A8F98"
INK = "#ECECEC"
config.background_color = BG
DATA = json.loads((Path(__file__).parent / "video_data.json").read_text())


def T(s, size=30, color=INK, **kw):
    return Text(s, font="Helvetica Neue", font_size=size, color=color, **kw)


def footer(*lines):
    g = VGroup(*[Text(l, font="Helvetica Neue", font_size=13, color=GREY_) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
    rule = Line(LEFT * 6.9, RIGHT * 6.9, stroke_width=1, color="#3a3e46")
    return VGroup(rule, g).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_edge(DOWN, buff=0.18).to_edge(LEFT, buff=0.3)


def heading(s):
    return T(s, 34, YEL).to_edge(UP, buff=0.45)


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
        self.boosting()
        self.ensemble()
        self.losses()
        self.interval()
        self.agaci()
        self.honesty()
        self.results()
        self.outro()

    # ── 1 ────────────────────────────────────────────────
    def title(self):
        route = VMobject(color=BLUE_, stroke_width=5).set_points_smoothly(
            [LEFT * 6.5 + DOWN * 2.2, LEFT * 3 + DOWN * 1.4, ORIGIN + DOWN * 2.3, RIGHT * 3 + DOWN * 1.5, RIGHT * 6.5 + DOWN * 2.1])
        bus = RoundedRectangle(width=0.45, height=0.22, corner_radius=0.05, fill_color=YEL, fill_opacity=1, stroke_width=0)
        t = T("When will my bus come?", 56, YEL).shift(UP * 0.6)
        s = T("the v4 arrival-time estimator behind the live bus maps", 28, GREY_).next_to(t, DOWN, buff=0.35)
        self.play(Create(route), Write(t), run_time=1.5)
        self.play(FadeIn(s, shift=UP * 0.2), MoveAlongPath(bus, route), run_time=2.5, rate_func=linear)
        self.wait(0.5)
        self.clear_all()

    # ── 2 ────────────────────────────────────────────────
    def problem(self):
        pts = [LEFT * 6 + DOWN * 0.8, LEFT * 3.6 + UP * 0.7, LEFT * 1 + DOWN * 0.1, RIGHT * 1.6 + UP * 1.1, RIGHT * 4 + DOWN * 0.4, RIGHT * 6 + UP * 0.4]
        route = VMobject(color=BLUE_, stroke_width=6).set_points_smoothly(pts)
        al = [0.0, 0.18, 0.36, 0.55, 0.74, 0.9, 1.0]
        stops = VGroup(*[Dot(route.point_from_proportion(a), radius=0.09, color=INK) for a in al])
        self.play(Create(route), LaggedStart(*[GrowFromCenter(d) for d in stops], lag_ratio=0.1), run_time=1.6)
        bus = RoundedRectangle(width=0.5, height=0.26, corner_radius=0.06, fill_color=YEL, fill_opacity=1, stroke_width=0)
        bus.move_to(route.point_from_proportion(0.25))
        k = stops[4]
        ring = Circle(radius=0.22, color=TEAL, stroke_width=4).move_to(k)
        lk = MathTex(r"\text{stop } k", color=TEAL, font_size=30).next_to(k, DOWN, buff=0.3)
        self.play(FadeIn(bus, scale=0.5), Create(ring), Write(lk))
        q = MathTex(r"t_k", "=", "t_0", "+", "y_k", font_size=50).to_edge(UP, buff=0.6)
        q[0].set_color(TEAL); q[2].set_color(YEL)
        brace = Brace(q[4], DOWN, color=GREY_)
        bt = T("what we predict", 20, GREY_).next_to(brace, DOWN, buff=0.1)
        self.play(Write(q))
        self.play(GrowFromCenter(brace), FadeIn(bt))
        tr = ValueTracker(0.25)
        bus.add_updater(lambda m: m.move_to(route.point_from_proportion(tr.get_value())))
        eta = always_redraw(lambda: T(f"{max(0, (0.74 - tr.get_value()) * 40):.1f} min", 24, YEL).next_to(bus, UP, buff=0.2))
        self.add(eta)
        self.play(tr.animate.set_value(0.5), run_time=3, rate_func=linear)
        note = T("GPS fixes come every 10 s (Delhi) to ~3 min (Chennai), late and noisy", 22, GREY_).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(note))
        self.wait(1)
        bus.clear_updaters()
        self.clear_all()

    # ── 3 ────────────────────────────────────────────────
    def nowcast(self):
        self.play(Write(heading("1 · the physics baseline: the nowcast")))
        xs = [-5.5, -3.3, -1.4, 0.8, 3.0, 5.4]
        cols = [TEAL, YEL, RED_, YEL, TEAL]
        segs = VGroup(*[Line([xs[i], 2.1, 0], [xs[i + 1], 2.1, 0], stroke_width=10, color=cols[i]) for i in range(5)])
        vl = VGroup(*[MathTex(f"v_{i + 1}", font_size=30, color=cols[i]).next_to(segs[i], UP, buff=0.12) for i in range(5)])
        self.play(Create(segs), Write(vl), run_time=1.2)
        f = MathTex(r"\hat y_k^{\,nc}", "=", r"\sum_{i\le k}", r"\frac{d_i}{", r"v_i", r"\;p(\Delta t)", "}", font_size=50).shift(UP * 0.75)
        self.play(Write(f), run_time=1.5)
        b1 = Brace(f[4], DOWN, color=BLUE_)
        v_txt = MathTex(r"v_i:\ \text{live (15 min)}\ \to\ \text{this hour, 7 days}\ \to\ \text{city}", font_size=28, color=BLUE_).next_to(b1, DOWN, buff=0.12)
        self.play(GrowFromCenter(b1), Write(v_txt))
        pace = MathTex(r"p(\Delta t) = 1 + (\pi_{\text{bus}} - 1)\,e^{-\Delta t/170\,\mathrm{s}}", font_size=32, color=YEL).next_to(v_txt, DOWN, buff=0.35)
        self.play(Indicate(f[5], color=YEL), Write(pace))
        ax = Axes(x_range=[0, 900, 300], y_range=[1, 1.4, 0.2], x_length=4, y_length=1.2, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2}).next_to(pace, DOWN, buff=0.25).shift(LEFT * 2)
        c = ax.plot(lambda t: 1 + 0.35 * math.exp(-t / 170), color=YEL)
        xl = MathTex(r"\Delta t", font_size=24, color=GREY_).next_to(ax.x_axis, RIGHT, buff=0.1)
        cap = T("a fast bus stays fast for a minute or two;\nafter that the road decides", 18, GREY_).next_to(ax, RIGHT, buff=0.6)
        self.play(Create(ax), Create(c), FadeIn(xl), FadeIn(cap))
        self.play(FadeIn(footer("Every bus is a traffic probe for the others: speeds pooled across all routes on a segment (Σ distance ÷ Σ time, stops included).",
                                "Kinematic waves: Lighthill & Whitham, Proc. R. Soc. A 1955; Richards, Operations Research 1956.")))
        self.wait(2.5)
        self.clear_all()

    # ── 4 ────────────────────────────────────────────────
    def residual(self):
        self.play(Write(heading("2 · learn what the physics gets wrong")))
        ax = Axes(x_range=[0, 30, 10], y_range=[0, 30, 10], x_length=4.6, y_length=4.6, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2}).shift(LEFT * 3.4 + DOWN * 0.15)
        xl = T("predicted, min", 18, GREY_).next_to(ax.x_axis, DOWN, buff=0.3)
        yl = T("actual, min", 18, GREY_).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.3)
        diag = DashedLine(ax.c2p(0, 0), ax.c2p(30, 30), color=GREY_)
        nums = VGroup(*[T(str(v), 14, GREY_).next_to(ax.c2p(v, 0), DOWN, buff=0.08) for v in (10, 20, 30)],
                      *[T(str(v), 14, GREY_).next_to(ax.c2p(0, v), LEFT, buff=0.08) for v in (10, 20, 30)])
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), Create(diag), FadeIn(nums))
        S = DATA["sample"]
        dots = VGroup(*[Dot(ax.c2p(min(b, 30), min(y, 30)), radius=0.035, color=BLUE_, fill_opacity=0.7) for y, b, p in S])
        lab = T("nowcast", 24, BLUE_).next_to(ax, UP, buff=0.1)
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.004), FadeIn(lab), run_time=1.6)
        f1 = MathTex(r"r = \log\frac{y}{\hat y^{\,nc}}", font_size=46, color=TEAL).shift(RIGHT * 3.2 + UP * 1.5)
        f2 = MathTex(r"\hat y = \hat y^{\,nc}\cdot e^{\hat r(\mathbf{x})}", font_size=46, color=YEL).next_to(f1, DOWN, buff=0.55)
        self.play(Write(f1))
        self.play(TransformFromCopy(f1, f2))
        moved = VGroup(*[Dot(ax.c2p(min(p, 30), min(y, 30)), radius=0.035, color=YEL, fill_opacity=0.75) for y, b, p in S])
        self.play(Transform(dots, moved), Transform(lab, T("v4", 24, YEL).move_to(lab)), run_time=2.2)
        why = VGroup(T("a ratio: one model for 1 and 30 min ahead", 20, GREY_),
                     T("unsure model → r̂ ≈ 0 → back to the physics", 20, GREY_),
                     T("dots: 400 real MTC predictions, held-out 23 Nov", 18, GREY_)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(f2, DOWN, buff=0.5)
        self.play(FadeIn(why, lag_ratio=0.2))
        self.play(FadeIn(footer("Residual learning on top of a routing estimate: Hu et al., DeeprETA (Uber), arXiv:2206.02127, 2022.",
                                "Wang et al., Learning to Estimate the Travel Time (WDR, DiDi), KDD 2018.")))
        self.wait(2.5)
        self.clear_all()

    # ── 5 ────────────────────────────────────────────────
    def features(self):
        h = heading("3 · the 74 things it looks at")
        self.play(Write(h))
        groups = [
            ("Baselines", TEAL, ["nowcast, 7-day history", "live corridor estimate", "live ÷ history ratio"]),
            ("Live traffic", BLUE_, ["corridor speed 5/15/30 min", "variability, stop-and-go", "city-wide speed"]),
            ("This bus", YEL, ["pace vs model", "speed since last fix", "time & km into trip"]),
            ("Road ahead", RED_, ["distance, stops ahead", "bends, turning angle", "OSM signals, flyovers, rail"]),
            ("When", PURP, ["time of day, weekday", "holidays, festivals", "rain 1–24 h, heat"]),
            ("Spacing", GREY_, ["headway to the bus ahead", "position on route", "route length"]),
        ]
        cards = VGroup()
        for name, col, items in groups:
            body = VGroup(*[T(it, 17) for it in items]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            card = VGroup(T(name, 24, col), body).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            cards.add(VGroup(SurroundingRectangle(card, buff=0.18, color=col, stroke_width=2, corner_radius=0.12), card))
        cards.arrange_in_grid(rows=2, cols=3, buff=(0.35, 0.35)).scale_to_fit_width(12.4).next_to(h, DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.15), run_time=2.2)
        self.wait(1.5)
        x = MathTex(r"\mathbf{x} = (x_1, x_2, \dots, x_{74})", font_size=52).move_to(cards)
        self.play(ReplacementTransform(cards, x), run_time=1.3)
        self.play(FadeIn(footer("Path geometry & context features: Fang et al., ConSTGAT (Baidu Maps), KDD 2020.  Bus pace & dwell: Cats, Transportmetrica B 2019.",
                                "Every feature uses only what was known at the moment of the prediction (the fleet replayed in time order).")))
        self.wait(1.5)
        self.clear_all()

    # ── 6 ────────────────────────────────────────────────
    def boosting(self):
        self.play(Write(heading("4 · gradient-boosted trees")))

        def tree(col):
            n0 = Dot(radius=0.08, color=col)
            n1, n2 = Dot(radius=0.07, color=col).shift(DL * 0.6), Dot(radius=0.07, color=col).shift(DR * 0.6)
            leaves = VGroup(*[Square(0.14, color=col, fill_opacity=0.8).move_to(p) for p in
                              (n1.get_center() + DL * 0.4, n1.get_center() + DR * 0.4, n2.get_center() + DL * 0.4, n2.get_center() + DR * 0.4)])
            edges = VGroup(Line(n0.get_center(), n1.get_center()), Line(n0.get_center(), n2.get_center()),
                           *[Line((n1 if i < 2 else n2).get_center(), l.get_center()) for i, l in enumerate(leaves)]).set_stroke(col, width=2)
            return VGroup(edges, n0, n1, n2, leaves)

        trees = VGroup(*[tree(c) for c in (BLUE_, TEAL, PURP, BLUE_, TEAL)]).arrange(RIGHT, buff=0.7).scale(0.85).shift(UP * 0.9 + LEFT * 0.4)
        plus = VGroup(*[MathTex("+", font_size=40).move_to((trees[i].get_right() + trees[i + 1].get_left()) / 2) for i in range(4)])
        dots_ = MathTex(r"\cdots", font_size=40).next_to(trees, RIGHT, buff=0.3)
        self.play(LaggedStart(*[Create(t) for t in trees], lag_ratio=0.25), FadeIn(plus), FadeIn(dots_), run_time=2.5)
        f = MathTex(r"F_M(\mathbf{x}) = \sum_{m=1}^{M} \eta\, f_m(\mathbf{x}),\qquad f_m \approx -\frac{\partial L(r,\,F_{m-1})}{\partial F_{m-1}}", font_size=40).shift(DOWN * 1.1)
        self.play(Write(f), run_time=1.6)
        cap = T("each small tree fixes what the trees before it still get wrong", 22, GREY_).next_to(f, DOWN, buff=0.4)
        self.play(FadeIn(cap))
        self.play(FadeIn(footer("Friedman, Greedy Function Approximation: A Gradient Boosting Machine, Annals of Statistics, 2001.")))
        self.wait(2)
        self.clear_all()

    # ── 7 ────────────────────────────────────────────────
    def ensemble(self):
        self.play(Write(heading("5 · three ensembles vote")))
        x = MathTex(r"\mathbf{x}", font_size=52).shift(LEFT * 5.2 + UP * 0.3)
        names = [("LightGBM", "leaf-wise", BLUE_, "0.40"), ("XGBoost", "depth-wise", TEAL, "0.50"), ("CatBoost", "symmetric", RED_, "0.10")]
        boxes = VGroup()
        for n, sub, c, w in names:
            b = RoundedRectangle(width=3.0, height=1.0, corner_radius=0.15, color=c, stroke_width=3)
            boxes.add(VGroup(b, VGroup(T(n, 26, c), T(sub + " trees", 16, GREY_)).arrange(DOWN, buff=0.06).move_to(b)))
        boxes.arrange(DOWN, buff=0.35).shift(LEFT * 1.2 + UP * 0.3)
        out = MathTex(r"\hat r", font_size=52, color=YEL).shift(RIGHT * 3.6 + UP * 0.3)
        ain = VGroup(*[Arrow(x.get_right(), b.get_left(), buff=0.15, stroke_width=3, color=GREY_) for b in boxes])
        aout = VGroup(*[Arrow(b.get_right(), out.get_left(), buff=0.15, stroke_width=3, color=GREY_) for b in boxes])
        ws = VGroup(*[MathTex(w, font_size=30, color=c).move_to(a.point_from_proportion(0.45) + UP * 0.25) for (_, _, c, w), a in zip(names, aout)])
        self.play(Write(x))
        self.play(LaggedStart(*[GrowArrow(a) for a in ain], lag_ratio=0.1), LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.15))
        self.play(LaggedStart(*[GrowArrow(a) for a in aout], lag_ratio=0.1), FadeIn(ws), Write(out))
        f = MathTex(r"\hat r = 0.40\,f_{\mathrm{LGB}}(\mathbf{x}) + 0.50\,f_{\mathrm{XGB}}(\mathbf{x}) + 0.10\,f_{\mathrm{Cat}}(\mathbf{x})", font_size=40, color=YEL).to_edge(DOWN, buff=1.35)
        self.play(Write(f))
        self.play(FadeIn(footer("Ke et al., LightGBM, NeurIPS 2017.  Chen & Guestrin, XGBoost, KDD 2016.  Prokhorenkova et al., CatBoost, NeurIPS 2018.",
                                "Weights picked on a held-out validation day. Trained on 5 systems: Chennai MTC, Bengaluru BMTC, Delhi, Mumbai, Gurugram.")))
        self.wait(2.5)
        self.clear_all()

    # ── 8 ────────────────────────────────────────────────
    def losses(self):
        self.play(Write(heading("robust losses: one strange trip can't drag the rest")))
        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 3, 1], x_length=6, y_length=3.6, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2}).shift(LEFT * 3 + DOWN * 0.3)
        el = MathTex(r"e = r - \hat r", font_size=26, color=GREY_).next_to(ax.x_axis, DOWN, buff=0.15)
        d = 1.0
        sq = ax.plot(lambda e: 0.5 * e * e, x_range=[-2.45, 2.45], color=GREY_, stroke_width=3)
        hub = ax.plot(lambda e: 0.5 * e * e if abs(e) <= d else d * (abs(e) - 0.5 * d), x_range=[-3, 3], color=BLUE_, stroke_width=5)
        ph = ax.plot(lambda e: d * d * ((1 + (e / d) ** 2) ** 0.5 - 1), x_range=[-3, 3], color=TEAL, stroke_width=4)
        mae = ax.plot(lambda e: abs(e) * 0.75, x_range=[-3, 3], color=RED_, stroke_width=3)
        self.play(Create(ax), FadeIn(el), Create(sq))
        self.play(Create(hub), Create(ph), Create(mae), run_time=1.6)
        lg = VGroup(
            MathTex(r"\tfrac12 e^2 \;\;(\text{not used})", font_size=30, color=GREY_),
            MathTex(r"\text{Huber: } \begin{cases}\tfrac12 e^2 & |e|\le\delta\\ \delta\,(|e|-\tfrac12\delta) & \text{else}\end{cases}", font_size=30, color=BLUE_),
            MathTex(r"\text{Pseudo-Huber: } \delta^2\!\left(\sqrt{1+(e/\delta)^2}-1\right)", font_size=30, color=TEAL),
            MathTex(r"\text{MAE: } |e|", font_size=30, color=RED_),
            MathTex(r"\delta = 0.25 \;\;(\approx \pm 28\%\text{ of the time})", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(RIGHT * 3.4 + DOWN * 0.2)
        self.play(FadeIn(lg, lag_ratio=0.2), run_time=1.6)
        self.play(FadeIn(footer("Huber, Robust Estimation of a Location Parameter, Annals of Mathematical Statistics, 1964.",
                                "LightGBM → Huber, XGBoost → Pseudo-Huber, CatBoost → MAE: different error shapes, so their mistakes partly cancel.")))
        self.wait(2.5)
        self.clear_all()

    # ── 9 ────────────────────────────────────────────────
    def interval(self):
        self.play(Write(heading("6 · an honest range: “8 in 10 buses arrive within …”")))
        nl = NumberLine(x_range=[0, 14, 2], length=10, color=GREY_).shift(UP * 1.0)
        nums = VGroup(*[T(str(v), 20, GREY_).next_to(nl.n2p(v), DOWN, buff=0.15) for v in range(0, 15, 2)])
        unit = T("minutes", 18, GREY_).next_to(nl, RIGHT)
        self.play(Create(nl), FadeIn(nums), FadeIn(unit))
        ax = Axes(x_range=[0, 14, 2], y_range=[0, 0.3, 0.1], x_length=10, y_length=1.6, tips=False,
                  axis_config={"stroke_opacity": 0}).move_to(nl, aligned_edge=DOWN)
        dens = lambda t: 0.28 * math.exp(-((math.log(max(t, 0.05)) - math.log(8.2)) ** 2) / (2 * 0.2 ** 2))
        curve = ax.plot(dens, x_range=[0.3, 14], color=TEAL, stroke_width=3)
        area = ax.get_area(curve, x_range=[6.3, 10.9], color=TEAL, opacity=0.35)
        self.play(Create(curve), FadeIn(area))
        lab = T("6–11 min", 24, TEAL).next_to(area, UP, buff=0.1)
        self.play(Write(lab))
        f1 = MathTex(r"\Big[\;\hat y^{\,nc} e^{\,q_{10}(\mathbf{x}) - m_b}\;,\;\; \hat y^{\,nc} e^{\,q_{90}(\mathbf{x}) + m_b}\;\Big]", font_size=42).shift(DOWN * 0.7)
        self.play(Write(f1), run_time=1.4)
        b = Brace(f1, DOWN, color=GREY_)
        bt = VGroup(MathTex(r"q_{10}, q_{90}:\ \text{LightGBM quantile models}", font_size=28, color=GREY_),
                    MathTex(r"m_b:\ \text{conformal margin per horizon } b\ (<\!3,\ 3\text{--}5,\ 5\text{--}10,\ 10\text{--}15,\ 15^{+}\text{ min})", font_size=28, color=GREY_)).arrange(DOWN, buff=0.12).next_to(b, DOWN, buff=0.12)
        self.play(GrowFromCenter(b), FadeIn(bt))
        self.play(FadeIn(footer("Romano, Patterson & Candès, Conformalized Quantile Regression, NeurIPS 2019.",
                                "Vovk, Gammerman & Shafer, Algorithmic Learning in a Random World (Mondrian conformal prediction), Springer 2005.")))
        self.wait(2.5)
        self.clear_all()

    # ── 10 ───────────────────────────────────────────────
    def agaci(self):
        self.play(Write(heading("…that keeps itself honest while it runs")))
        f = MathTex(r"m^{(\gamma)}_b \;\leftarrow\; m^{(\gamma)}_b + \gamma\,\big(\mathbf{1}[\text{arrival outside}] - 0.20\big)", font_size=40, color=TEAL).shift(UP * 2.0)
        g = MathTex(r"\gamma \in \{0.002,\,0.005,\,0.015,\,0.05\},\qquad m_b = \mathrm{median}_\gamma\, m^{(\gamma)}_b", font_size=32, color=GREY_).next_to(f, DOWN, buff=0.25)
        self.play(Write(f), run_time=1.5)
        self.play(FadeIn(g))
        A = DATA["agaci"]
        n = len(A["cov_adaptive"])
        ax = Axes(x_range=[0, n, max(1, n // 4)], y_range=[0.70, 0.90, 0.05], x_length=8.5, y_length=2.8, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2}).shift(DOWN * 1.0)
        yn = VGroup(*[T(f"{int(round(v * 100))}%", 16, GREY_).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0.70, 0.75, 0.80, 0.85, 0.90)])
        tgt = DashedLine(ax.c2p(0, 0.8), ax.c2p(n, 0.8), color=YEL)
        tl = T("target 80%", 16, YEL).next_to(tgt, RIGHT, buff=0.1)
        self.play(Create(ax), FadeIn(yn), Create(tgt), FadeIn(tl))
        clip = lambda v: min(0.9, max(0.7, v))
        st = ax.plot_line_graph(list(range(n)), [clip(v) for v in A["cov_static"]], line_color=GREY_, add_vertex_dots=False, stroke_width=3)
        ad = ax.plot_line_graph(list(range(n)), [clip(v) for v in A["cov_adaptive"]], line_color=TEAL, add_vertex_dots=False, stroke_width=4)
        lg = VGroup(T("fixed margin", 18, GREY_), T("adaptive (AgACI)", 18, TEAL)).arrange(RIGHT, buff=0.6).next_to(ax, UP, buff=0.1).align_to(ax, RIGHT)
        self.play(Create(st), Create(ad), FadeIn(lg), run_time=3, rate_func=linear)
        cap = T(f"running coverage on {A['n']:,} real MTC predictions 15+ min ahead (23 Nov); live, the margins track each city", 17, GREY_).next_to(ax, DOWN, buff=0.2)
        self.play(FadeIn(cap))
        self.play(FadeIn(footer("Gibbs & Candès, Adaptive Conformal Inference Under Distribution Shift, NeurIPS 2021.",
                                "Zaffran, Dieuleveut, Féron, Goude & Josse, Adaptive Conformal Predictions for Time Series (AgACI), ICML 2022.")))
        self.wait(2.5)
        self.clear_all()

    # ── 11 ───────────────────────────────────────────────
    def honesty(self):
        self.play(Write(heading("checking it honestly")))
        days = VGroup()
        for lab, c in [("train", BLUE_)] * 5 + [("val", TEAL), ("test", YEL)]:
            r = RoundedRectangle(width=1.4, height=0.7, corner_radius=0.1, color=c, fill_opacity=0.15)
            days.add(VGroup(r, T(lab, 20, c).move_to(r)))
        days.arrange(RIGHT, buff=0.15).shift(UP * 1.4)
        arrow = Arrow(days.get_left() + DOWN * 0.6, days.get_right() + DOWN * 0.6, buff=0, color=GREY_, stroke_width=3)
        tl = T("time →", 18, GREY_).next_to(arrow, DOWN, buff=0.08)
        self.play(LaggedStart(*[FadeIn(d) for d in days], lag_ratio=0.08), GrowArrow(arrow), FadeIn(tl))
        pts = VGroup(
            T("• whole days held out; the test day is never seen in training", 24),
            T("• the fleet replayed fix by fix: a prediction sees only the past", 24),
            T("• history speeds rebuilt from the 7 days before the test day", 24),
            T("   (the first published scores used a history model that contained it)", 20, GREY_),
            T("• then a live shadow run against buses' real arrivals", 24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).next_to(tl, DOWN, buff=0.45)
        self.play(FadeIn(pts, lag_ratio=0.2), run_time=2.2)
        self.wait(2.5)
        self.clear_all()

    # ── 12 ───────────────────────────────────────────────
    def results(self):
        self.play(Write(heading("live, 24 Nov 2026 evening: mean error, stops 1–30 min ahead")))
        data = [("Chennai MTC", 198, 158, 104), ("Delhi", 199, 190, 126), ("Bengaluru BMTC", 351, 94, 84)]
        ax = Axes(x_range=[0, 3, 1], y_range=[0, 360, 60], x_length=9.5, y_length=4.0, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2}).shift(DOWN * 0.45 + RIGHT * 0.4)
        yn = VGroup(*[T(str(v), 16, GREY_).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in range(0, 361, 60)])
        yl = T("seconds", 18, GREY_).rotate(PI / 2).next_to(yn, LEFT, buff=0.15)
        self.play(Create(ax), FadeIn(yn), FadeIn(yl))
        grow, labels = [], VGroup()
        for i, (city, shown, nc, v4) in enumerate(data):
            for j, (val, col) in enumerate([(shown, GREY_), (nc, BLUE_), (v4, YEL)]):
                x = i + 0.5 + (j - 1) * 0.26
                b = Rectangle(width=0.22, height=ax.y_axis.unit_size * val, fill_color=col, fill_opacity=0.9, stroke_width=0)
                b.move_to(ax.c2p(x, 0), aligned_edge=DOWN)
                grow.append(GrowFromEdge(b, DOWN))
                labels.add(T(str(val), 16, col).next_to(b, UP, buff=0.05))
            labels.add(T(city, 20).next_to(ax.c2p(i + 0.5, 0), DOWN, buff=0.2))
        leg = VGroup(*[VGroup(Square(0.18, fill_color=c, fill_opacity=1, stroke_width=0), T(t, 18, c)).arrange(RIGHT, buff=0.12)
                       for c, t in ((GREY_, "what the map showed before"), (BLUE_, "nowcast"), (YEL, "v4"))]).arrange(RIGHT, buff=0.5).next_to(ax, UP, buff=0.15).align_to(ax, RIGHT)
        self.play(LaggedStart(*grow, lag_ratio=0.08), FadeIn(leg), run_time=2.4)
        self.play(FadeIn(labels))
        self.play(FadeIn(footer("Same buses, same moments, same stops for all three. Held-out days offline vs nowcast: MTC −32 %, BMTC −35 %, Delhi −35 %;",
                                "Mumbai only −6 % once its history model was rebuilt properly. The 80 % range covered 77–84 % of real arrivals live.")))
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
