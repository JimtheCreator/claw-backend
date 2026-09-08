"""Mobile presentation: one plan, one conditional path, no detector heatmap."""
from html import escape
from math import isfinite
from textwrap import wrap

import pandas as pd
import plotly.graph_objects as go

PRESENTATION_VERSION = "first-leg-v1"
NEXT_MOVE_PRESENTATION_VERSION = "next-move-v1"


def presentation_version(plan):
    return NEXT_MOVE_PRESENTATION_VERSION if plan.get("execution_policy") == "next_move" else PRESENTATION_VERSION


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
        self.next_move = self.plan.get("execution_policy") == "next_move"
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
                          margin=dict(l=45, r=110, t=220, b=285), showlegend=False,
                          xaxis_rangeslider_visible=False,
                          meta={"presentation_version": PRESENTATION_VERSION})
        bounds = [float(self.visible.low.min()), float(self.visible.high.max())]
        if self.next_move and self.plan.get("decision_level"):
            bounds.append(self.plan["decision_level"]["price"])
        for s in self.scenarios:
            keys = ("trigger", "target", "invalidation") if self.next_move else ("trigger",)
            bounds.extend(float(s[k]) for k in keys if s.get(k) is not None)
        if self.next_move:
            bounds.extend(t["price"] for t in self.plan.get("targets", []))
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
                        "NEXT LEVEL · TIMING UNSPECIFIED" if self.next_move else "FIRST LEG · PROJECTION ENDS AT LEVEL",
                        self.muted, yref="y domain", size=18)
        fig.add_trace(go.Candlestick(x=self.visible.timestamp.astype(str), open=self.visible.open,
                                    high=self.visible.high, low=self.visible.low, close=self.visible.close,
                                    increasing=dict(line=dict(color=self.green, width=2), fillcolor=self.green),
                                    decreasing=dict(line=dict(color=self.red, width=2), fillcolor=self.red), name="Price"))
        fig.add_trace(go.Scatter(x=[self.now], y=[self.current], mode="markers",
                                marker=dict(color="white", size=9), name="Last close"))
        self.draw_evidence()
        self.draw_scenario()
        self.draw_headings()
        return fig

    def annotation(self, x, y, text, color, *, yref="y", size=23, **kwargs):
        self.fig.add_annotation(x=x, y=y, xref="x", yref=yref, text=text, showarrow=False,
                                font=dict(size=size, color=color), **kwargs)

    def draw_evidence(self):
        """Plot price-located facts; summarize every material fact below."""
        start, now = pd.Timestamp(self.start), pd.Timestamp(self.now)
        located = []
        for item in self.plan.get("chart_evidence", []):
            if not item.get("plot", True):
                continue
            timestamp = pd.Timestamp(item["timestamp"])
            if start.tzinfo is None:
                timestamp = timestamp.tz_localize(None)
            elif timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(start.tzinfo)
            if start <= timestamp <= now:
                located.append(item)
        self.omitted_anchor_count = max(0, len(located) - 5)
        for i, evidence in enumerate(located[:5]):
            timestamp = pd.Timestamp(evidence["timestamp"])
            # ChartEngine can supply naive UTC datetimes; compare consistently.
            if start.tzinfo is None:
                timestamp = timestamp.tz_localize(None)
            elif timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(start.tzinfo)
            reference = evidence.get("reference_timestamp")
            if reference:
                ref = pd.Timestamp(reference)
                if start.tzinfo is None:
                    ref = ref.tz_localize(None)
                elif ref.tzinfo is None:
                    ref = ref.tz_localize(start.tzinfo)
                self.fig.add_shape(type="line",x0=max(start,ref),x1=timestamp,
                                   y0=evidence["price"],y1=evidence["price"],
                                   line=dict(color=self.amber,width=2,dash="dot"))
            self.fig.add_annotation(x=timestamp, y=evidence["price"], xref="x", yref="y",
                                    text=f"<b>{chr(65+i)}. {escape(evidence['kind'])}</b>",
                                    showarrow=True, arrowhead=2, arrowcolor=self.amber,
                                    ax=-50-(i % 3)*115, ay=-45-(i // 3)*55,
                                    bgcolor=self.background,
                                    font=dict(size=18, color=self.amber))
            zone = evidence.get("zone")
            if zone:
                self.fig.add_shape(type="rect", x0=timestamp, x1=self.now, y0=zone["bottom"], y1=zone["top"],
                                   line=dict(color=self.amber, width=1), fillcolor="rgba(255,208,120,0.04)", layer="below")

    def draw_scenario(self):
        if self.next_move:
            self.draw_next_move()
            return
        self.draw_first_leg()

    def draw_first_leg(self):
        """Display only the existing scenario's approach to its trigger.

        Do not change planner eligibility or reinterpret a pending entry as an
        approved trade in the opposite direction. The scenario remains intact.
        """
        if not self.scenarios:
            self.annotation(self.now + (self.end - self.now) / 2, 0.5,
                            "<b>No supported path yet</b><br>Wait for confirmed structure<br>and complete context.",
                            self.amber, yref="y domain", size=25)
            return
        s = self.scenarios[0]
        trigger = s["trigger"]
        bullish = trigger > self.current
        color = self.green if bullish else self.red
        duration = self.end - self.now
        endpoint = self.now + duration * .85
        self.fig.add_trace(go.Scatter(x=[self.now, endpoint], y=[self.current, trigger],
                                     mode="lines+markers", name="Conditional scenario",
                                     line=dict(color=color, width=5, dash="dash"),
                                     marker=dict(size=12, color=color)))
        self.fig.add_shape(type="line", x0=self.now, x1=self.end, y0=trigger, y1=trigger,
                           line=dict(color=color, width=1, dash="dot"))
        self.annotation(self.end, trigger, f"Next level {price(trigger)}", color,
                        xanchor="right", yshift=28, bgcolor=self.background)

    def draw_headings(self):
        if self.next_move:
            self.draw_next_headings()
            return
        p = self.plan
        interval = "1 month" if p.get("interval") == "1M" else p.get("interval", "")
        mtfa = p.get("evidence", {}).get("mtfa", {})
        htf = " · ".join(f"{tf} {trend or 'unconfirmed'}" for tf, trend in mtfa.get("htf_trends", {}).items()) if mtfa.get("enabled") is True else ""
        context = f"HTF: {htf}" if htf else "MTFA ON" if mtfa.get("enabled") else "MTFA OFF"
        title = (f"{'UP' if self.scenarios[0]['trigger'] > self.current else 'DOWN'} toward {price(self.scenarios[0]['trigger'])}"
                 if self.scenarios else "Wait for a valid scenario")
        self.fig.update_layout(title=dict(text=f"<b>{escape(self.analysis.get('symbol', ''))} · {escape(interval)} chart</b>"
                                              f"<br><span style='font-size:23px'>Local: {escape(p.get('trend_direction', 'undetermined'))} · {escape(context)}</span>",
                                          x=0.03, y=0.95, xanchor="left", yanchor="top", font=dict(size=34)))
        reason = p.get("reason") or p.get("context_summary") or ""
        if self.scenarios:
            reason = reason.replace(". Retest entry is still pending.", ".")
        self.fig.add_annotation(x=0, y=1.22, xref="paper", yref="paper", xanchor="left", yanchor="top", align="left",
                                text=f"<b>{'NEXT PROJECTED MOVE' if self.scenarios else 'WAIT'} · {escape(title)}</b><br>" + lines(reason, 116),
                                showarrow=False, font=dict(size=23, color=self.amber))
        if self.scenarios:
            s = self.scenarios[0]
            text = lines(f"Projection ends at {price(s['trigger'])}. Reassess there. This is the existing scenario's first leg; timing and arrival are unconfirmed.", 120)
        else:
            text = lines(p.get("reason") or "No confirmed structural levels. No entry or forecast.", 112)
        self.fig.add_annotation(x=0, y=-0.13, xref="paper", yref="paper", xanchor="left", yanchor="top",
                                align="left", text=text, showarrow=False, font=dict(size=22, color="#ecf1f8"))
        evidence = self.plan.get("chart_evidence", [])
        icons = {"passed":"✓", "failed":"✕", "unavailable":"?"}
        facts = [f"{icons.get(e.get('status'),'•')} {e['label']}" +
                 (" [required]" if e.get("mandatory") and e.get("status") != "passed" else "") for e in evidence]
        why = "Decision facts: " + " · ".join(facts) if facts else "Decision facts: none available"
        if getattr(self, "omitted_anchor_count", 0):
            why += f" · 5 chart anchors shown; {self.omitted_anchor_count} additional located fact(s) summarized here"
        self.fig.add_annotation(x=0, y=-0.26, xref="paper", yref="paper", xanchor="left", yanchor="top",
                                text=lines(why, 150), showarrow=False, font=dict(size=16, color=self.muted))
        self.fig.add_annotation(x=0, y=-0.46, xref="paper", yref="paper", xanchor="left", yanchor="top",
                                text=f"Latest {len(self.visible)} of {len(self.candles)} candles · Observed facts are separate from the illustrative pending path · Experimental rules · {PRESENTATION_VERSION}",
                                showarrow=False, font=dict(size=15, color=self.muted))

    def draw_next_move(self):
        """A qualified move ends at its exit; WAIT has no approach arrow."""
        p = self.plan
        level = p.get("decision_level")
        if p.get("action") not in {"long", "short"} or not self.scenarios:
            wait_position = .5
            if level:
                y = level["price"]
                lo,hi=self.fig.layout.yaxis.range
                wait_position = .72 if (y-lo)/(hi-lo)<.5 else .28
                self.fig.add_shape(type="line", x0=self.now, x1=self.end, y0=y, y1=y,
                                   line=dict(color=self.amber,width=2,dash="dot"))
                self.annotation(self.end,y,f"Watch {price(y)}",self.amber,xanchor="right",yshift=22,
                                bgcolor=self.background)
            self.annotation(self.now+(self.end-self.now)*.5,wait_position,
                            "<b>WAIT</b><br>No entry confirmed<br>No approach forecast",self.amber,
                            yref="y domain",size=24)
            return
        s = self.scenarios[0]
        color = self.green if p["action"] == "long" else self.red
        end = self.now+(self.end-self.now)*.88
        self.fig.add_trace(go.Scatter(x=[self.now,end],y=[s["trigger"],s["target"]],
            mode="lines+markers",name="Next move to exit",line=dict(color=color,width=5,dash="dash"),
            marker=dict(size=11,color=color)))
        labels=[(s["target"],f"EXIT 100% · {price(s['target'])}",color),
                (s["trigger"],f"Entry reference {price(s['trigger'])}",self.muted),
                (s["invalidation"],f"STOP {price(s['invalidation'])}",self.red)]
        lo,hi=self.fig.layout.yaxis.range
        previous=-100
        for value,label,tint in sorted(labels):
            natural=(value-lo)/(hi-lo)*395
            placed=max(natural,previous+32);previous=placed
            self.fig.add_shape(type="line",x0=self.now,x1=self.end,y0=value,y1=value,
                               line=dict(color=tint,width=1,dash="dot"))
            self.annotation(self.end,value,label,tint,xanchor="right",yshift=placed-natural,
                            bgcolor=self.background,size=21)

    def draw_next_headings(self):
        p=self.plan
        interval="1 month" if p.get("interval")=="1M" else p.get("interval","")
        mtfa=p.get("evidence",{}).get("mtfa",{})
        htf=" · ".join(f"{tf} {trend or 'unconfirmed'}" for tf,trend in mtfa.get("htf_trends",{}).items()) if mtfa.get("enabled") is True else ""
        context=f"HTF: {htf}" if htf else "MTFA ON" if mtfa.get("enabled") else "MTFA OFF"
        self.fig.update_layout(meta={"presentation_version":presentation_version(p)},
            title=dict(text=f"<b>{escape(self.analysis.get('symbol',''))} · {escape(interval)} chart</b>"
                f"<br><span style='font-size:23px'>Local: {escape(p.get('trend_direction','undetermined'))} · {escape(context)}</span>",
                x=.03,y=.96,xanchor="left",yanchor="top",font=dict(size=32)))
        tint=self.green if p.get("action")=="long" else self.red if p.get("action")=="short" else self.amber
        self.fig.add_annotation(x=0,y=1.23,xref="paper",yref="paper",xanchor="left",yanchor="top",align="left",
            text=lines(p.get("reason","WAIT"),108),showarrow=False,font=dict(size=22,color=tint))
        note=("Entry uses the last closed candle. Recheck live price and costs before entry. Exit at the marked level; reassess there."
              if p.get("action") in {"long","short"} else "The marked level is a place to reassess. It does not establish a trade toward it.")
        self.fig.add_annotation(x=0,y=-.13,xref="paper",yref="paper",xanchor="left",yanchor="top",align="left",
            text=lines(note,130),showarrow=False,font=dict(size=18,color="#ecf1f8"))
        icons={"passed":"✓","failed":"✕","unavailable":"?"}
        facts=[f"{icons.get(e.get('status'),'•')} {e['label']}"+
               (" [required]" if e.get("mandatory") and e.get("status")!="passed" else "")
               for e in p.get("chart_evidence",[])]
        ledger="Decision facts: "+" · ".join(facts) if facts else "Decision facts: no confirmed setup"
        if getattr(self,"omitted_anchor_count",0):
            ledger+=f" · 5 chart anchors shown; {self.omitted_anchor_count} more summarized here"
        self.fig.add_annotation(x=0,y=-.27,xref="paper",yref="paper",xanchor="left",yanchor="top",align="left",
            text=lines(ledger,160),showarrow=False,font=dict(size=16,color=self.muted))
        self.fig.add_annotation(x=0,y=-.67,xref="paper",yref="paper",xanchor="left",yanchor="top",
            text=f"Last {len(self.visible)} of {len(self.candles)} closed candles · Dashed line is an illustrative route, not a timing forecast · next-move-v1",
            showarrow=False,font=dict(size=14,color=self.muted))
