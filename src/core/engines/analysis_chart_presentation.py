"""Mobile presentation: one plan, one conditional path, no detector heatmap."""
from html import escape
from math import isfinite
from textwrap import wrap

import pandas as pd
import plotly.graph_objects as go

PRESENTATION_VERSION = "conditional-forecast-v8"
NEXT_MOVE_PRESENTATION_VERSION = "next-move-v3"


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
        self.market_read = self.plan.get("market_read") or {}
        # Keep distant context in the caption instead of crushing candle scale.
        self.context_levels = {side: item for side in ("support", "resistance")
                               if (item := self.market_read.get(side))
                               and abs(item["price"]-self.current) <= 3*self.span}

    def build_scenarios(self):
        # Fail closed for rejected or legacy scenario-only plans as well as new
        # output. Valid geometry alone cannot authorize a directional forecast.
        direction = {"long": "bullish", "short": "bearish"}.get(self.plan.get("action"))
        if (direction is None or self.plan.get("market_context") not in {"local", "aligned"}
                or (self.plan.get("setup_quality") or {}).get("eligible") is not True):
            return []
        scenario = self.plan.get("primary_scenario") or self.plan.get("forecast_scenario")
        if not scenario or scenario.get("setup") is not True or scenario.get("direction") != direction:
            return []
        levels = [scenario.get(key) for key in ("trigger", "invalidation", "target")]
        if scenario.get("direction") not in {"bullish", "bearish"} or levels[0] is None or levels[1] is None:
            return []
        try:
            invalid = any(not isfinite(float(v)) or float(v) <= 0 for v in levels if v is not None)
        except (ValueError, TypeError):
            return []
        if invalid:
            return []
        return [{**scenario, **{k: float(scenario[k]) if scenario.get(k) is not None else None
                               for k in ("trigger", "target", "invalidation")}}]

    def complete_forecast(self):
        if not self.scenarios:
            return False
        s = self.scenarios[0]
        target = s.get("target")
        if target is None:
            return False
        return (s["invalidation"] < s["trigger"] < target if s["direction"] == "bullish"
                else target < s["trigger"] < s["invalidation"])

    def reference_forecast_issue(self):
        """Do not draw a current-price reward box backwards or past invalidation."""
        if not self.complete_forecast():
            return "Target / invalidation unavailable<br>No complete forecast can be drawn"
        s = self.scenarios[0]
        bullish = s["direction"] == "bullish"
        if (self.current <= s["invalidation"] if bullish else self.current >= s["invalidation"]):
            return "Latest close is beyond scenario invalidation<br>Reassess; no active forecast"
        if (self.current >= s["target"] if bullish else self.current <= s["target"]):
            return "Latest close has reached or passed the target<br>Reassess; no remaining forecast"
        return None

    def figure(self):
        fig = self.fig = go.Figure()
        fig.update_layout(template="plotly_dark", width=1600, height=900,
                          paper_bgcolor=self.background, plot_bgcolor=self.background,
                          font=dict(family="Arial, sans-serif", size=23, color="#ecf1f8"),
                          margin=dict(l=45, r=110, t=195, b=250), showlegend=False,
                          xaxis_rangeslider_visible=False,
                          meta={"presentation_version": PRESENTATION_VERSION})
        bounds = [float(self.visible.low.min()), float(self.visible.high.max())]
        if not self.next_move:
            bounds.extend(item["price"] for item in self.context_levels.values())
        if self.next_move and self.plan.get("decision_level"):
            bounds.append(self.plan["decision_level"]["price"])
        for s in self.scenarios:
            keys = ("trigger", "target", "invalidation") if self.next_move or self.complete_forecast() else ("trigger",)
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
                        "NEXT LEVEL · TIMING UNSPECIFIED" if self.next_move else
                        "CONDITIONAL FORECAST · TIMING NOT ESTIMATED" if not self.reference_forecast_issue() else
                        "MARKET WATCH · NO FORECAST APPROVED",
                        self.muted, yref="y domain", size=18)
        fig.add_trace(go.Candlestick(x=self.visible.timestamp.astype(str), open=self.visible.open,
                                    high=self.visible.high, low=self.visible.low, close=self.visible.close,
                                    increasing=dict(line=dict(color=self.green, width=2), fillcolor=self.green),
                                    decreasing=dict(line=dict(color=self.red, width=2), fillcolor=self.red), name="Price"))
        fig.add_trace(go.Scatter(x=[self.now], y=[self.current], mode="markers",
                                marker=dict(color="white", size=9), name="Last close"))
        self.draw_evidence()
        if not self.next_move:
            self.draw_market_read()
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
        evidence_items = list(self.plan.get("chart_evidence", []))
        observed = self.market_read.get("last_break")
        if observed and not any(e.get("timestamp") == observed["timestamp"] and e.get("kind") == observed["kind"] for e in evidence_items):
            evidence_items.insert(0, observed)
        for item in evidence_items:
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
        self.draw_forecast()

    def draw_market_read(self):
        for side, item in self.context_levels.items():
            y = item["price"]
            self.fig.add_shape(type="line", x0=self.start, x1=self.now, y0=y, y1=y,
                               line=dict(color=self.muted, width=1, dash="dot"))
            self.annotation(self.start, y, f"{side.title()} {price(y)}", self.muted,
                            xanchor="left", yshift=-18 if side == "support" else 18,
                            size=19, bgcolor=self.background)

    def draw_direction_badge(self, direction):
        """Reference-style Long/Short candle badge; never a projection line."""
        bullish = direction == "bullish"
        fill = "#00bd83" if bullish else "#ff4e64"
        pending = not self.next_move or self.plan.get("wait_for_confirmation") is not False
        text = ("Long pending ▲" if bullish else "Short pending ▼") if pending else ("Long ▲" if bullish else "Short ▼")
        lo, hi = self.fig.layout.yaxis.range
        layout = self.fig.layout
        width = layout.width - layout.margin.l - layout.margin.r
        height = layout.height - layout.margin.t - layout.margin.b
        xmin, xmax = (pd.Timestamp(x) for x in layout.xaxis.range)
        x = (self.now-xmin)/(xmax-xmin)
        candle = self.visible.iloc[-1]
        anchor = float(candle.low if bullish else candle.high)
        sign = -1 if bullish else 1
        tip = (anchor-lo)/(hi-lo) + sign*6/height
        near, far = tip+sign*6/height, tip+sign*32/height
        half_width = 80 if pending else 43
        left, right, notch = x-half_width/width, x+half_width/width, 6/width
        self.fig.add_shape(type="path", name="Forecast direction badge",
                           xref="x domain", yref="y domain", fillcolor=fill, line_width=0,
                           path=f"M {x},{tip} L {x+notch},{near} L {right},{near} L {right},{far} L {left},{far} L {left},{near} L {x-notch},{near} Z")
        self.fig.add_annotation(name="Forecast direction label", x=self.now,
                                y=lo+(near+far)/2*(hi-lo), xref="x", yref="y",
                                text=f"<b>{text}</b>", showarrow=False,
                                font=dict(size=18, color="white"))

    def draw_forecast_price_tags(self, labels):
        """Left-pointing price flags, with leaders preserving exact price anchors.

        Pixel-spaced tags can separate almost identical TP/confirmation levels
        without moving their prices or changing the actual reward/risk geometry.
        """
        lo, hi = self.fig.layout.yaxis.range
        height = self.fig.layout.height - self.fig.layout.margin.t - self.fig.layout.margin.b
        placed, previous = [], -100
        for value, text, fill, key in sorted(labels):
            natural = (value-lo)/(hi-lo)*height
            center = max(natural, previous+34, 18)
            placed.append((value, text, fill, key, natural, center))
            previous = center
        overflow = max(0, placed[-1][-1] - (height-18))
        for value, text, fill, key, natural, center in placed:
            y, actual = (center-overflow)/height, natural/height
            half = 13/height
            # A short angled leader is necessary only when tags would overlap.
            self.fig.add_shape(type="line", name=f"Forecast {key} leader",
                               xref="x domain", yref="y domain",
                               x0=.745, x1=.77, y0=actual, y1=y,
                               line=dict(color=fill, width=1.5))
            self.fig.add_shape(type="path", name=f"Forecast {key} tag",
                               xref="x domain", yref="y domain",
                               path=f"M .77,{y} L .78,{y+half} L .985,{y+half} L .985,{y-half} L .78,{y-half} Z",
                               fillcolor=fill, line_width=0)
            self.fig.add_annotation(name=f"Forecast {key} label", x=.785, y=y,
                                    xref="x domain", yref="y domain", xanchor="left", yanchor="middle",
                                    text=f"<b>{escape(text)}</b>", showarrow=False,
                                    font=dict(size=18, color="white"))

    def draw_forecast(self):
        """Visualize supplied conditional direction, target and invalidation.

        Future x coordinates are layout space, NOT predicted arrival times.
        Shading and the illustrative direction start at the last closed candle.
        This reference is NOT an entry; the separate activation and original
        confirmation requirements remain unchanged. No fabricated retest legs.
        """
        if not self.scenarios:
            self.annotation(self.now + (self.end - self.now) / 2, 0.5,
                            (f"<b>{escape(self.market_read.get('trend_direction', 'undetermined').title())} structure</b><br>Entry criteria not met<br>See next check below."
                             if self.market_read else "<b>No supported forecast</b><br>Missing structural levels<br>No entry confirmed."),
                            self.amber, yref="y domain", size=25)
            return
        s = self.scenarios[0]
        trigger = s["trigger"]
        issue = self.reference_forecast_issue()
        if issue:
            self.fig.add_shape(type="line", x0=self.now, x1=self.end, y0=trigger, y1=trigger,
                               line=dict(color=self.amber, width=2, dash="dot"))
            self.annotation(self.end, trigger, f"Activation {price(trigger)}", self.amber,
                            xanchor="right", yshift=25, bgcolor=self.background)
            self.annotation(self.now+(self.end-self.now)*.5, .83,
                            issue,
                            self.amber, yref="y domain", size=20)
            return
        # Current-candle reference geometry, not the pending-entry risk ratio.
        x0, x1 = self.now, self.end-self.step*2
        target, stop = s["target"], s["invalidation"]
        # Exactly two future-only bands, not an overlay on every detector zone.
        for y, fill, name in ((target, "rgba(66,223,178,0.22)", "Forecast reward"),
                              (stop, "rgba(255,115,131,0.22)", "Forecast risk")):
            self.fig.add_shape(type="rect", name=name, x0=x0, x1=x1,
                               y0=min(self.current, y), y1=max(self.current, y),
                               line_width=0, fillcolor=fill, layer="below")
        self.draw_direction_badge(s["direction"])
        self.fig.update_layout(meta={"presentation_version": PRESENTATION_VERSION,
                                    "forecast_reference": "last_closed_candle",
                                    "forecast_reference_price": self.current,
                                    "activation_price": trigger})
        pending_setup = s.get("setup") and self.plan.get("action") in {"long", "short"}
        label = "Entry pending" if pending_setup else "Confirm"
        labels = [(target, f"TP {price(target)}", "#127650", "TP"),
                  (trigger, f"{label} {price(trigger)}", "#946014", "confirmation"),
                  (self.current, f"NOW {price(self.current)} · ref.", "#355575", "reference"),
                  (stop, f"SL / invalid. {price(stop)}", "#ae3548", "SL")]
        for value, tint in ((target, self.green), (stop, self.red),
                            (self.current, self.muted), (trigger, self.amber)):
            self.fig.add_shape(type="line", x0=x0, x1=x1, y0=value, y1=value,
                               line=dict(color=tint, width=1, dash="solid"),
                               opacity=0.6, layer="below")
        self.draw_forecast_price_tags(labels)
        risk, reward = abs(trigger-stop), abs(target-trigger)
        self.forecast_rr = reward/risk
        # Extend activation back to the observed trigger pivot when known.
        origin = s.get("trigger_timestamp")
        if origin:
            origin = pd.Timestamp(origin)
            if self.now.tzinfo is None:
                origin = origin.tz_localize(None)
            elif origin.tzinfo is None:
                origin = origin.tz_localize(self.now.tzinfo)
            if origin <= self.now:
                self.fig.add_shape(type="line", x0=max(self.start, origin), x1=x0, y0=trigger, y1=trigger,
                                   line=dict(color=self.amber, width=1, dash="dot"))

    def draw_headings(self):
        if self.next_move:
            self.draw_next_headings()
            return
        p = self.plan
        interval = "1 month" if p.get("interval") == "1M" else p.get("interval", "")
        mtfa = p.get("evidence", {}).get("mtfa", {})
        htf = " · ".join(f"{tf} {trend or 'unconfirmed'}" for tf, trend in mtfa.get("htf_trends", {}).items()) if mtfa.get("enabled") is True else ""
        context = f"HTF: {htf}" if htf else "MTFA ON" if mtfa.get("enabled") else "MTFA OFF"
        pending_setup = bool(self.scenarios and self.scenarios[0].get("setup")
                             and p.get("action") in {"long", "short"})
        visible_forecast = not self.reference_forecast_issue()
        status = ((f"{self.scenarios[0]['direction'].upper()} FORECAST · " +
                   ("ENTRY PENDING" if pending_setup else "NO ENTRY APPROVED")) if visible_forecast else
                  f"{self.market_read['trend_direction'].upper()} STRUCTURE · NO ENTRY CONFIRMED" if self.market_read else "WAIT · NO ENTRY CONFIRMED")
        title = (f"TP {price(self.scenarios[0]['target'])}" if visible_forecast else
                 "Scenario needs reassessment" if self.scenarios else "Forecast withheld")
        self.fig.update_layout(title=dict(text=f"<b>{escape(self.analysis.get('symbol', ''))} · {escape(interval)} chart</b>"
                                              f"<br><span style='font-size:23px'>Local: {escape(p.get('trend_direction', 'undetermined'))} · {escape(context)}</span>",
                                          x=0.03, y=0.95, xanchor="left", yanchor="top", font=dict(size=34)))
        reason = p.get("reason") or p.get("context_summary") or ""
        self.fig.add_annotation(x=0, y=1.20, xref="paper", yref="paper", xanchor="left", yanchor="top", align="left",
                                text=f"<b>{escape(status)} · {escape(title)}</b><br>" + lines(reason, 116),
                                showarrow=False, font=dict(size=23, color=self.amber))
        if self.scenarios:
            s = self.scenarios[0]
            confirmation = s.get("confirmation") or p.get("confirmation_required") or "Require a closed-candle trigger before entry."
            reference = (f"Activation-based R:R {self.forecast_rr:.2f}R before fees. " if hasattr(self, "forecast_rr") else "")
            if hasattr(self, "forecast_rr") and self.forecast_rr < 1.5:
                reference += "Below the planner's 1.5R minimum; not a qualifying trade. "
            scope = "Local scenario only; HTF context is not validated. " if s.get("basis") == "local_structure" and mtfa.get("enabled") else ""
            text = lines(f"Condition: {confirmation} {scope}{reference}TP/SL are scenario levels, not placed orders.", 125)
        else:
            watch = p.get("structure_watch") or {}
            next_check = watch.get("confirmation") or self.market_read.get("next_check")
            text = lines("Next check: " + next_check, 112) if next_check else lines(p.get("reason") or "No confirmed structural levels. No entry or forecast.", 112)
        self.fig.add_annotation(x=0, y=-0.13, xref="paper", yref="paper", xanchor="left", yanchor="top",
                                align="left", text=text, showarrow=False, font=dict(size=22, color="#ecf1f8"))
        evidence = self.plan.get("chart_evidence", [])
        icons = {"passed":"✓", "failed":"✕", "unavailable":"?"}
        facts = [f"{icons.get(e.get('status'),'•')} {e['label']}" +
                 (" [required]" if e.get("mandatory") and e.get("status") != "passed" else "") for e in evidence]
        why = "Entry checklist: " + " · ".join(facts) if facts else "Entry checklist not reached; see blocking reason above."
        if self.market_read:
            levels = [f"{side.title()} {price(item['price'])}" + (" (outside view)" if side not in self.context_levels else "")
                      for side in ("support", "resistance") if (item := self.market_read.get(side))]
            observed = self.market_read.get("last_break")
            if observed:
                levels.append(f"Last {observed['direction']} {observed['kind']}: {observed['bars_ago']} bars ago")
            if levels:
                why = "Local context: " + " · ".join(levels) + " · " + why
        if getattr(self, "omitted_anchor_count", 0):
            why += f" · 5 chart anchors shown; {self.omitted_anchor_count} additional located fact(s) summarized here"
        self.fig.add_annotation(x=0, y=-0.36, xref="paper", yref="paper", xanchor="left", yanchor="top",
                                text=lines(why, 150), showarrow=False, font=dict(size=16, color=self.muted))
        forecast_note = ("Pending badge requires the stated retest; it is not an entry now · Shading from latest close is illustrative"
                         if visible_forecast else "Observed structure only · No directional forecast approved")
        self.fig.add_annotation(x=0, y=-0.49, xref="paper", yref="paper", xanchor="left", yanchor="top",
                                text=f"Latest {len(self.visible)} of {len(self.candles)} candles · {forecast_note} · {PRESENTATION_VERSION}",
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
        self.draw_direction_badge("bullish" if p["action"] == "long" else "bearish")
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
            text=f"Last {len(self.visible)} of {len(self.candles)} closed candles · Long/Short badge marks direction; timing is unspecified · {NEXT_MOVE_PRESENTATION_VERSION}",
            showarrow=False,font=dict(size=14,color=self.muted))
