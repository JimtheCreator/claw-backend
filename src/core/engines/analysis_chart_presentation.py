"""Mobile presentation: one plan, one conditional path, no detector heatmap."""
from html import escape
from math import isfinite
from textwrap import wrap

import pandas as pd
import plotly.graph_objects as go

PRESENTATION_VERSION = "scenario-v3"


def price(value):
    return f"{value:,.2f}" if abs(value) >= 10 else f"{value:.6g}"


def lines(value, width=105):
    return "<br>".join(escape(line) for line in wrap(str(value), width))


class AnalysisChartPresentation:
    green, red, amber = "#42dfb2", "#ff7383", "#ffd078"
    muted, background = "#a4afc1", "#101722"

    def __init__(self, candles, analysis, smc):
        self.candles, self.analysis, self.smc = candles, analysis, smc
        self.plan = analysis["trade_plan"]
        self.visible = candles.tail(60)
        self.start, self.now = self.visible.timestamp.iloc[0], self.visible.timestamp.iloc[-1]
        deltas = self.visible.timestamp.diff().dropna()
        self.step = deltas[deltas > pd.Timedelta(0)].median() if len(deltas) else pd.NaT
        if pd.isna(self.step):
            self.step = pd.Timedelta(minutes=1)
        self.end = self.now + self.step * 42
        self.current = float(self.visible.close.iloc[-1])
        self.span = max(float(self.visible.high.max() - self.visible.low.min()), abs(self.current) * 0.0001, 1e-8)
        self.scenarios = self.build_scenarios()

    def build_scenarios(self):
        # The decision layer owns forecasts. Never draw an unrelated breakout
        # guess just because the result says WAIT.
        scenario = self.plan.get("primary_scenario")
        if not scenario:
            return []
        levels = [scenario.get(key) for key in ("trigger", "invalidation", "target")]
        if scenario.get("direction") not in {"bullish", "bearish"} or levels[0] is None or levels[1] is None:
            return []
        if any(not isfinite(float(v)) for v in levels if v is not None):
            return []
        return [dict(scenario)]

    def figure(self):
        fig = self.fig = go.Figure()
        fig.update_layout(template="plotly_dark", width=1600, height=900,
                          paper_bgcolor=self.background, plot_bgcolor=self.background,
                          font=dict(family="Arial, sans-serif", size=23, color="#ecf1f8"),
                          margin=dict(l=45, r=110, t=220, b=180), showlegend=False,
                          xaxis_rangeslider_visible=False,
                          meta={"presentation_version": PRESENTATION_VERSION})
        bounds = [float(self.visible.low.min()), float(self.visible.high.max())]
        for s in self.scenarios:
            bounds.extend(float(s[k]) for k in ("trigger", "target", "invalidation") if s.get(k) is not None)
        padding = max(max(bounds) - min(bounds), self.span) * 0.19
        fig.update_yaxes(range=[min(bounds) - padding, max(bounds) + padding], autorange=False,
                         side="right", nticks=6, tickformat=",.2f" if self.current >= 10 else ".6g",
                         gridcolor="#263142", zeroline=False)
        fig.update_xaxes(type="date", range=[self.start - self.step, self.end + self.step * 2],
                         nticks=6, gridcolor="#202b3b", tickfont=dict(size=20),
                         tickformat="%H:%M\n%d %b" if self.step < pd.Timedelta(days=1) else "%d %b\n%Y")
        fig.add_shape(type="rect", x0=self.now, x1=self.end + self.step * 2, y0=0, y1=1,
                      xref="x", yref="y domain", line_width=0, fillcolor="rgba(102,151,240,0.07)", layer="below")
        fig.add_shape(type="line", x0=self.now, x1=self.now, y0=0, y1=1,
                      xref="x", yref="y domain", line=dict(color=self.muted, width=1, dash="dot"))
        self.annotation(self.now, 1.035, "NOW", self.muted, yref="y domain", size=18)
        self.annotation(self.now + (self.end - self.now) * 0.58, 1.035,
                        "CONDITIONAL PATH · NOT A PREDICTION", self.muted, yref="y domain", size=18)
        fig.add_trace(go.Candlestick(x=self.visible.timestamp.astype(str), open=self.visible.open,
                                    high=self.visible.high, low=self.visible.low, close=self.visible.close,
                                    increasing=dict(line=dict(color=self.green, width=2), fillcolor=self.green),
                                    decreasing=dict(line=dict(color=self.red, width=2), fillcolor=self.red), name="Price"))
        fig.add_trace(go.Scatter(x=[self.now], y=[self.current], mode="markers",
                                marker=dict(color="white", size=9), name="Last close"))
        self.draw_scenario()
        self.draw_headings()
        return fig

    def annotation(self, x, y, text, color, *, yref="y", size=23, **kwargs):
        self.fig.add_annotation(x=x, y=y, xref="x", yref=yref, text=text, showarrow=False,
                                font=dict(size=size, color=color), **kwargs)

    def draw_scenario(self):
        if not self.scenarios:
            self.annotation(self.now + (self.end - self.now) / 2, 0.5,
                            "<b>No supported path yet</b><br>Wait for confirmed structure<br>and complete context.",
                            self.amber, yref="y domain", size=25)
            return
        s = self.scenarios[0]
        bullish, setup = s["direction"] == "bullish", s.get("setup", False)
        color = self.green if bullish else self.red
        trigger, target, invalidation = s["trigger"], s.get("target"), s["invalidation"]
        duration = self.end - self.now
        t1, t2, t3 = [self.now + duration * f for f in (0.25, 0.51, 0.9)]
        # Overshoot illustrates a candle close, not an additional objective.
        overshoot = min(self.span * 0.06, abs(target - trigger) * 0.2) if target is not None else self.span * 0.06
        confirmed = trigger + (overshoot if bullish else -overshoot)
        xs, ys = [self.now, t1, t2], [self.current, confirmed, trigger]
        if target is not None:
            xs.append(t3)
            ys.append(target)
        self.fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Conditional scenario",
                                     line=dict(color=color, width=5, dash="dash"),
                                     marker=dict(size=12, color=color)))
        self.annotation(t1, confirmed, "1 · Close", color, yshift=28 if bullish else -28, bgcolor=self.background)
        self.annotation(t2, trigger, "2 · Retest", color, yshift=-28 if bullish else 28, bgcolor=self.background)
        if target is not None:
            self.fig.add_annotation(x=t3, y=target, ax=-28, ay=32 if bullish else -32,
                                    xref="x", yref="y", text="", showarrow=True, arrowhead=3,
                                    arrowsize=1.3, arrowwidth=4, arrowcolor=color)
        # Highlight only the selected execution zone, not every detected zone.
        zone = self.plan.get("entry_zone") if setup else None
        if zone:
            self.fig.add_shape(type="rect", x0=self.now - self.step * 5, x1=t2,
                               y0=zone["bottom"], y1=zone["top"], xref="x", yref="y",
                               line=dict(color=self.amber, width=1), fillcolor="rgba(255,208,120,0.08)", layer="below")
        labels = [(trigger, f"{'Entry' if setup else 'Confirm'} {price(trigger)}", color),
                  (invalidation, f"{'Stop' if setup else 'Invalid'} {price(invalidation)}", self.red)]
        if target is not None:
            labels.append((target, f"3 · Target {price(target)}", color))
        lo, hi = self.fig.layout.yaxis.range
        # Separate labels in pixel space without moving their real price levels.
        previous = -100
        for level, label, label_color in sorted(labels):
            natural = (level - lo) / (hi - lo) * 500
            placed = max(natural, previous + 34)
            previous = placed
            self.fig.add_shape(type="line", x0=self.now, x1=self.end, y0=level, y1=level,
                               line=dict(color=label_color, width=1, dash="dot"))
            self.annotation(self.end, level, label, label_color, xanchor="right",
                            yshift=placed - natural, bgcolor=self.background, borderpad=3)

    def draw_headings(self):
        p = self.plan
        interval = "1 month" if p.get("interval") == "1M" else p.get("interval", "")
        mtfa = p.get("evidence", {}).get("mtfa", {})
        htf = " · ".join(f"{tf} {trend or 'unconfirmed'}" for tf, trend in mtfa.get("htf_trends", {}).items())
        context = f"HTF: {htf}" if htf else "MTFA ON" if mtfa.get("enabled") else "MTFA OFF"
        title = self.scenarios[0]["title"] if self.scenarios else "Wait for a valid scenario"
        self.fig.update_layout(title=dict(text=f"<b>{escape(self.analysis.get('symbol', ''))} · {escape(interval)} chart</b>"
                                              f"<br><span style='font-size:23px'>Local: {escape(p.get('trend_direction', 'undetermined'))} · {escape(context)}</span>",
                                          x=0.03, y=0.95, xanchor="left", yanchor="top", font=dict(size=34)))
        self.fig.add_annotation(x=0, y=1.22, xref="paper", yref="paper", xanchor="left", yanchor="top", align="left",
                                text=f"<b>WAIT FOR CONFIRMATION · {escape(title)}</b><br>" + lines(p.get("reason") or p.get("context_summary") or "", 116),
                                showarrow=False, font=dict(size=23, color=self.amber))
        if self.scenarios:
            s = self.scenarios[0]
            direction = "above" if s["direction"] == "bullish" else "below"
            confirmation = f"Close {direction} {price(s['trigger'])}; retest must hold. "
            confirmation += "Conditional entry only after rejection + structure confirmation." if s.get("setup") else "No entry or stop approved yet; reassess risk after confirmation."
            if s.get("extra_confirmation"):
                confirmation += " " + s["extra_confirmation"]
            alternative = f"If {price(s['invalidation'])} breaks first, cancel this scenario and reassess."
            if s.get("target") is None:
                alternative += " No unswept target mapped; projection stops at the retest."
            text = "<b>1 → 2:</b> " + lines(confirmation, 113) + "<br><b>Invalidation:</b> " + lines(alternative, 113)
        else:
            text = lines(p.get("reason") or "No confirmed structural levels. No entry or forecast.", 112)
        self.fig.add_annotation(x=0, y=-0.13, xref="paper", yref="paper", xanchor="left", yanchor="top",
                                align="left", text=text, showarrow=False, font=dict(size=22, color="#ecf1f8"))
        self.fig.add_annotation(x=0, y=-0.29, xref="paper", yref="paper", xanchor="left", yanchor="top",
                                text=f"Latest {len(self.visible)} of {len(self.candles)} candles · Path timing illustrative, not a price/time guarantee · {PRESENTATION_VERSION}",
                                showarrow=False, font=dict(size=17, color=self.muted))
